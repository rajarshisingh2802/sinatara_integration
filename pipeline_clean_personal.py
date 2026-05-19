"""
SINTARA V3 — Multi-Model Prompt Generation & Evaluation Engine
===============================================================

Architecture:
- Generators: GPT-5.1, Gemini 2.5 Flash, (optional) Claude Sonnet
- Judge: Claude Haiku (2-stage evaluation with 6-factor scoring)
- UI: Real-time scoreboard with colored output
- Storage: Supabase for analytics, in-memory cache for deduplication
- Pricing: USD/INR cost tracking per API call

Usage:
    python pipeline_clean_personal.py                # Silver (2 generators)
    python pipeline_clean_personal.py --diamond      # Diamond (3 generators)

Author: Your Name
Version: 3.0 (Updated with GPT-5.1)
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import anthropic
import google.generativeai as genai
import openai

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

from scoreboard_ui import ScoreboardUI, LiveProgressBar, Colors

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# Exchange rate for USD to INR conversion
USD_TO_INR = 95.7

# Retry limits for API calls and JSON parsing
MAX_RETRIES_JSON = 3  # Retry JSON parsing up to 3 times
MAX_RETRIES_API = 2   # Retry API calls up to 2 times

# ==============================================================================
# COST TABLE (per MILLION tokens)
# ==============================================================================
# Format: "model_name": (input_price_per_M, output_price_per_M)
# Updated May 2026 with latest OpenAI/Google pricing
#
# GPT-5.1: Significantly more expensive than GPT-4.1 but much better quality
# Gemini 3: Super cheap option, excellent for cost optimization
# Claude Haiku: Budget judge, used for evaluation to keep costs down
# Claude Sonnet: Premium option for Diamond tier (3 generators)

_COST_TABLE = {
    "gpt-5.1": (15.0, 60.0),                          # $15/M in, $60/M out
    "gemini-3-flash": (0.075, 0.30),                  # $0.075/M in, $0.30/M out
    "claude-haiku-4-5": (1.0, 5.0),                   # $1/M in, $5/M out
    "claude-sonnet-4-6": (3.0, 15.0),                 # $3/M in, $15/M out
}

# Approximation: 0.75 words per token (used when API doesn't return token count)
_WORDS_PER_TOKEN = 0.75

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def _estimate_tokens(text: str) -> int:
    """
    Estimate token count using word-to-token ratio.
    
    Used as fallback when API doesn't return actual token counts.
    Formula: words / 0.75 ≈ tokens
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count (minimum 1)
    """
    return max(1, int(len(text.split()) / _WORDS_PER_TOKEN))


def _cost_for_model(model: str, inp: int, out: int) -> float:
    """
    Calculate cost in USD for a single API call.
    
    Looks up model pricing from _COST_TABLE and calculates:
    cost = (input_tokens * input_rate / 1M) + (output_tokens * output_rate / 1M)
    
    Args:
        model: Model name (key in _COST_TABLE)
        inp: Input token count
        out: Output token count
        
    Returns:
        Cost in USD as float
        
    Example:
        >>> _cost_for_model("gpt-5.1", 500, 300)
        0.0255  # (500*15/1M) + (300*60/1M)
    """
    inp_rate, out_rate = _COST_TABLE.get(model, (1.0, 5.0))  # Default: Claude Haiku pricing
    return (inp * inp_rate / 1_000_000) + (out * out_rate / 1_000_000)


def _intent_hash(intent: str) -> str:
    """
    Create a unique fingerprint of user's intent for caching.
    
    Uses SHA-256 to create a deterministic hash. Same intent always 
    produces the same hash, enabling cache lookups.
    
    Args:
        intent: User's intent string
        
    Returns:
        64-character hex SHA-256 hash
        
    Example:
        >>> _intent_hash("Write a prompt")
        "a3f2c1d8e5b7f9a2c4e6g8h0i2j4k6l8m0n2o4p6"
    """
    return hashlib.sha256(intent.strip().lower().encode()).hexdigest()


# ==============================================================================
# DATA CONTAINERS (Dataclasses)
# ==============================================================================

@dataclass
class Message:
    """
    Single message in a conversation.
    
    Attributes:
        role: "user" or "assistant"
        content: The message text
    """
    role: str
    content: str


@dataclass
class ModelResponse:
    """
    Response from a model, with both raw and parsed versions.
    
    Attributes:
        raw: Unparsed response text from API
        parsed: Parsed JSON as Python dict (empty if no JSON)
    """
    raw: str
    parsed: dict = field(default_factory=dict)


@dataclass
class ViewPrompts:
    """
    Three prompt variants generated for one user intent.
    
    Attributes:
        optimistic: Positive perspective, emphasizes opportunity
        critical: Negative perspective, emphasizes risks
        creative: Novel perspective, lateral thinking
    """
    optimistic: str = ""
    critical: str = ""
    creative: str = ""

    def as_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        return {
            "optimistic": self.optimistic,
            "critical": self.critical,
            "creative": self.creative,
        }


@dataclass
class ScoreBreakdown:
    """
    Six quality scores for a prompt (1-10 each).
    
    Attributes:
        clarity: Are instructions clear?
        specificity: How specific to intent?
        actionability: How easy to execute?
        creativity: How novel/insightful?
        robustness: Handles edge cases?
        conciseness: Length appropriate?
    """
    clarity: int = 0
    specificity: int = 0
    actionability: int = 0
    creativity: int = 0
    robustness: int = 0
    conciseness: int = 0

    def as_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "clarity": self.clarity,
            "specificity": self.specificity,
            "actionability": self.actionability,
            "creativity": self.creativity,
            "robustness": self.robustness,
            "conciseness": self.conciseness,
        }

    def average(self) -> float:
        """Calculate average of all 6 scores."""
        scores = list(self.as_dict().values())
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class RunContext:
    """
    Tracks tokens and costs for a single pipeline run.
    
    All API calls within one pipeline execution add to these totals.
    Property cost_inr automatically converts USD to INR.
    
    Attributes:
        input_tokens: Total input tokens across all API calls
        output_tokens: Total output tokens across all API calls
        cost_usd: Total cost in USD
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def cost_inr(self) -> float:
        """Automatically convert USD cost to INR."""
        return self.cost_usd * USD_TO_INR

    def add_call(self, inp: int, out: int, cost: float):
        """
        Record one API call's token and cost usage.
        
        Args:
            inp: Input tokens used
            out: Output tokens used
            cost: Cost in USD
        """
        self.input_tokens += inp
        self.output_tokens += out
        self.cost_usd += cost


# Global context for current pipeline run
# Reset at start of each run_full_pipeline() call
_current_context = RunContext()


# ==============================================================================
# ABSTRACT BASE CLASS FOR ALL MODEL BACKENDS
# ==============================================================================

class ModelBase(abc.ABC):
    """
    Abstract base class for all AI model integrations.
    
    Subclasses implement:
    - __init__: Initialize API client and credentials
    - _call_model: Make actual API call to model
    
    Inherited methods handle JSON parsing, retry logic, fence stripping.
    """

    def send(
        self,
        messages: list[Message],
        *,
        system: str = ""
    ) -> ModelResponse:
        """
        Send messages to model, get plain text response.
        
        Args:
            messages: List of Message objects
            system: System prompt (optional)
            
        Returns:
            ModelResponse with raw text
        """
        raw = self._call_model(messages, system=system)
        return ModelResponse(raw=raw)

    def send_for_json(
        self,
        messages: list[Message],
        *,
        system: str = "",
        retries: int = MAX_RETRIES_JSON,
    ) -> ModelResponse:
        """
        Send messages, expect JSON response, parse it.
        
        Appends JSON-only instruction to system prompt and retries
        JSON parsing up to 'retries' times if it fails.
        
        Args:
            messages: List of Message objects
            system: System prompt (optional)
            retries: Max retry count (default: 3)
            
        Returns:
            ModelResponse with both raw and parsed JSON
            
        Raises:
            RuntimeError: If JSON parsing fails after all retries
        """

        # Strict instruction for JSON-only response
        json_instruction = (
            "Respond with valid JSON only. "
            "No markdown. No explanations. No preamble."
        )

        # Append instruction to system prompt
        json_system = (
            f"{system}\n\n{json_instruction}"
            if system else json_instruction
        )

        last_raw = ""

        # Retry loop: attempt up to retries+1 times
        for attempt in range(retries + 1):
            # Call the model
            raw = self._call_model(messages, system=json_system)
            last_raw = raw
            
            # Remove markdown code fences if present
            cleaned = self._strip_fences(raw)

            # Try to parse JSON
            try:
                parsed = json.loads(cleaned)
                return ModelResponse(raw=raw, parsed=parsed)
            except json.JSONDecodeError:
                # If not last attempt, try again
                if attempt < retries:
                    continue

        # All retries exhausted, raise error
        raise RuntimeError(
            f"JSON parsing failed after {retries + 1} attempts.\n"
            f"Raw output:\n{last_raw}"
        )

    @abc.abstractmethod
    def _call_model(
        self,
        messages: list[Message],
        *,
        system: str = ""
    ) -> str:
        """
        Make actual API call to model. Implemented by subclasses.
        
        Must:
        1. Call the model API
        2. Extract text response
        3. Get token counts (actual or estimated)
        4. Calculate cost using _cost_for_model()
        5. Add to global _current_context
        6. Return response text only
        """
        pass

    @staticmethod
    def _strip_fences(text: str) -> str:
        """
        Remove markdown code fence markers (``` symbols).
        
        Handles both ```json and ``` patterns.
        
        Args:
            text: Text potentially wrapped in code fences
            
        Returns:
            Text with fences removed
            
        Example:
            >>> _strip_fences("```json\\n{...}\\n```")
            "{...}"
        """
        text = text.strip()
        
        # Remove opening fence
        if text.startswith("```json"):
            text = text[7:]  # Remove ```json (7 chars)
        elif text.startswith("```"):
            text = text[3:]  # Remove ``` (3 chars)
        
        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3]  # Remove last 3 chars
            
        return text.strip()


# ==============================================================================
# CLAUDE BACKEND (Anthropic API)
# ==============================================================================

class ClaudeBackend(ModelBase):
    """
    Integration with Anthropic's Claude API.
    
    Used for:
    - Judging prompts (Claude Haiku 4.5) — cheap and reliable
    - Optional: Premium tier 3rd generator (Claude Sonnet 4.6)
    
    Env var required: ANTHROPIC_API_KEY
    """

    DEFAULT_MODEL = "claude-haiku-4-5"  # Budget model for judging
    DEFAULT_MAX_TOKENS = 2048

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize Claude API client.
        
        Args:
            api_key: API key (uses env var if not provided)
            model: Model name
            max_tokens: Max output length
        """
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def _call_model(
        self,
        messages: list[Message],
        *,
        system: str = ""
    ) -> str:
        """Make API call to Claude."""

        # Convert Message objects to API format
        api_msgs = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        # Build request kwargs
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=api_msgs,
        )

        # Add system prompt if provided
        if system:
            kwargs["system"] = system

        # Time the API call
        start = time.time()
        response = self._client.messages.create(**kwargs)
        duration = time.time() - start

        # Extract text from response
        text = response.content[0].text

        # Get token counts from response usage
        inp = getattr(
            response.usage,
            "input_tokens",
            _estimate_tokens(system + text)  # Fallback: estimate
        )

        out = getattr(
            response.usage,
            "output_tokens",
            _estimate_tokens(text)  # Fallback: estimate
        )

        # Calculate cost and add to global context
        cost = _cost_for_model(self.model, inp, out)
        _current_context.add_call(inp, out, cost)

        return text


# ==============================================================================
# OPENAI BACKEND (GPT API)
# ==============================================================================

class OpenAIBackend(ModelBase):
    """
    Integration with OpenAI's GPT API.
    
    Used for: Primary prompt generation (GPT-5.1)
    
    Env var required: OPENAI_API_KEY
    """

    DEFAULT_MODEL = "gpt-5.1"  # Latest and most expensive model
    DEFAULT_MAX_TOKENS = 2048

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """Initialize OpenAI API client."""
        self.model = model
        self.max_tokens = max_tokens
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )

    def _call_model(
        self,
        messages: list[Message],
        *,
        system: str = ""
    ) -> str:
        """Make API call to GPT."""

        # Build message list (OpenAI uses separate system message)
        api_msgs = []

        if system:
            api_msgs.append({
                "role": "system",
                "content": system
            })

        api_msgs += [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        # Time the API call
        start = time.time()
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=api_msgs,
        )
        duration = time.time() - start

        # Extract text
        text = response.choices[0].message.content or ""

        # Get token counts
        if response.usage:
            inp = response.usage.prompt_tokens
            out = response.usage.completion_tokens
        else:
            inp = _estimate_tokens(system + text)
            out = _estimate_tokens(text)

        # Calculate cost
        cost = _cost_for_model(self.model, inp, out)
        _current_context.add_call(inp, out, cost)

        return text


# ==============================================================================
# GEMINI BACKEND (Google API)
# ==============================================================================

class GeminiBackend(ModelBase):
    """
    Integration with Google's Gemini API.
    
    Used for: Secondary prompt generation (Gemini 2.5 Flash)
    Very cheap option for cost optimization.
    
    Env var required: GOOGLE_API_KEY
    """

    DEFAULT_MODEL = "gemini-3-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """Initialize Gemini API client."""
        self.model = model
        genai.configure(
            api_key=api_key or os.environ.get("GOOGLE_API_KEY")
        )
        self._client = genai.GenerativeModel(self.model)

    def _call_model(
        self,
        messages: list[Message],
        *,
        system: str = ""
    ) -> str:
        """Make API call to Gemini."""

        # Gemini uses flat content list (prepend system if provided)
        parts = []
        if system:
            parts.append(system)

        parts += [m.content for m in messages]

        # Time the API call
        start = time.time()
        response = self._client.generate_content("\n\n".join(parts))
        duration = time.time() - start

        # Extract text
        text = response.text

        # Get token counts (Gemini provides usage_metadata)
        if hasattr(response, 'usage_metadata'):
            inp = response.usage_metadata.input_token_count
            out = response.usage_metadata.output_token_count
        else:
            # Fallback: estimate if API doesn't provide
            inp = _estimate_tokens(system + text)
            out = _estimate_tokens(text)

        # Calculate cost
        cost = _cost_for_model(self.model, inp, out)
        _current_context.add_call(inp, out, cost)

        return text


# ==============================================================================
# GENERATOR CLASS
# ==============================================================================

class Generator(ModelBase):
    """
    Generates three prompt variants using a model backend.
    
    For each user intent, produces:
    - Optimistic: positive framing
    - Critical: negative framing
    - Creative: lateral thinking
    
    All three variants are complete, standalone prompts ready to use.
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
        You are an expert prompt engineer.
        
        Generate exactly THREE prompt variants based on the user's intent:
        - Optimistic: assumes best case, emphasizes opportunity
        - Critical: assumes worst case, emphasizes risks
        - Creative: thinks outside the box, novel approach
        
        Each variant should be a complete, standalone prompt that another AI could use.
        Make them distinct and complementary.
        
        Return JSON with these exact keys:
        {
          "optimistic": "...",
          "critical": "...",
          "creative": "..."
        }
    """)

    def __init__(
        self,
        model_backend: ModelBase,
        name: str = "",
        max_tokens: int = 500,
    ):
        """
        Initialize generator.
        
        Args:
            model_backend: ModelBase instance (Claude, GPT, Gemini)
            name: Display name (defaults to backend class name)
            max_tokens: Max output length
        """
        self._backend = model_backend
        self.name = name or type(model_backend).__name__
        self.max_tokens = max_tokens

    def _call_model(
        self,
        messages: list[Message],
        *,
        system: str = ""
    ) -> str:
        """Delegate to backend."""
        return self._backend._call_model(messages, system=system)

    def generate(self, intent: str) -> ViewPrompts:
        """
        Generate three prompt variants for the intent.
        
        Args:
            intent: What the user wants
            
        Returns:
            ViewPrompts with three variants
        """
        messages = [Message(role="user", content=intent)]
        response = self.send_for_json(
            messages,
            system=self.SYSTEM_PROMPT,
            retries=MAX_RETRIES_JSON,
        )

        return ViewPrompts(
            optimistic=response.parsed.get("optimistic", ""),
            critical=response.parsed.get("critical", ""),
            creative=response.parsed.get("creative", ""),
        )


# ==============================================================================
# JUDGE CLASS (Two-Stage Evaluation)
# ==============================================================================

class Judge(ModelBase):
    """
    Two-stage prompt evaluator using Claude Haiku.
    
    Stage 1 - Intra-view: Compare candidates within each view
    (optimistic vs optimistic, etc.) in parallel
    
    Stage 2 - Cross-view: Compare the 3 view winners to find the best overall
    
    Scoring: 6 factors (clarity, specificity, actionability, creativity, robustness, conciseness)
    Each factor rated 1-10.
    """

    INTRA_SYSTEM = textwrap.dedent("""\
        You are a rigorous prompt evaluator.
        
        Score each prompt on these 6 factors (scale 1-10):
        
        1. Clarity: How clear and unambiguous are the instructions?
        2. Specificity: How specific to the user's intent?
        3. Actionability: How easy for another AI to execute?
        4. Creativity: How novel, insightful, or clever?
        5. Robustness: How well does it handle edge cases and variations?
        6. Conciseness: Is the length appropriate? Not too verbose, not too terse?
        
        Return JSON:
        {
          "scores": {
            "clarity": int (1-10),
            "specificity": int (1-10),
            "actionability": int (1-10),
            "creativity": int (1-10),
            "robustness": int (1-10),
            "conciseness": int (1-10)
          },
          "winner_id": int (0 or 1),
          "reason": "brief explanation"
        }
    """)

    FINAL_SYSTEM = textwrap.dedent("""\
        You are a rigorous prompt evaluator.
        
        Three prompts compete, each representing a different perspective:
        - Optimistic
        - Critical
        - Creative
        
        Score each on these 6 factors (scale 1-10):
        
        1. Clarity: How clear and unambiguous are the instructions?
        2. Specificity: How specific to the user's intent?
        3. Actionability: How easy for another AI to execute?
        4. Creativity: How novel, insightful, or clever?
        5. Robustness: How well does it handle edge cases and variations?
        6. Conciseness: Is the length appropriate?
        
        Then pick the SINGLE BEST prompt overall, considering all factors.
        
        Return JSON:
        {
          "scores": {
            "clarity": int (1-10),
            "specificity": int (1-10),
            "actionability": int (1-10),
            "creativity": int (1-10),
            "robustness": int (1-10),
            "conciseness": int (1-10)
          },
          "winner_id": int (0, 1, or 2),
          "reason": "brief explanation"
        }
    """)

    def __init__(
        self,
        model_backend: Optional[ModelBase] = None,
        ui: Optional[ScoreboardUI] = None,
    ):
        """
        Initialize judge.
        
        Args:
            model_backend: Judge backend (defaults to Claude Haiku)
            ui: ScoreboardUI for display (optional)
        """
        self._backend = model_backend or ClaudeBackend()
        self.ui = ui

    def _call_model(
        self,
        messages: list[Message],
        *,
        system: str = ""
    ) -> str:
        """Delegate to backend."""
        return self._backend._call_model(messages, system=system)

    def _parse_scores(self, parsed: dict) -> ScoreBreakdown:
        """
        Extract scores from judge's JSON response.
        
        Defaults to 5 if any score is missing.
        """
        scores_dict = parsed.get("scores", {})
        return ScoreBreakdown(
            clarity=int(scores_dict.get("clarity", 5)),
            specificity=int(scores_dict.get("specificity", 5)),
            actionability=int(scores_dict.get("actionability", 5)),
            creativity=int(scores_dict.get("creativity", 5)),
            robustness=int(scores_dict.get("robustness", 5)),
            conciseness=int(scores_dict.get("conciseness", 5)),
        )

    def evaluate_view(
        self,
        view_name: str,
        candidates: list[str]
    ) -> tuple[str, str, ScoreBreakdown]:
        """
        Stage 1: Judge candidates within one view.
        
        Args:
            view_name: "optimistic", "critical", or "creative"
            candidates: Prompt variants from different generators
            
        Returns:
            (winning_prompt, reason, scores)
        """

        # If only one candidate, pick it by default
        if len(candidates) == 1:
            scores = ScoreBreakdown()
            return candidates[0], "Only one candidate available.", scores

        # Format candidates with IDs for judge to compare
        numbered = "\n\n".join(
            f"ID {i}:\n{candidate}"
            for i, candidate in enumerate(candidates)
        )

        user_msg = Message(
            role="user",
            content=f"VIEW: {view_name}\n\n{numbered}"
        )

        # Send to judge for evaluation
        response = self.send_for_json(
            [user_msg],
            system=self.INTRA_SYSTEM,
            retries=MAX_RETRIES_JSON,
        )

        # Extract winner ID and validate
        winner_id = response.parsed.get("winner_id", 0)

        if not isinstance(winner_id, int):
            winner_id = 0

        # Ensure winner_id is in valid range
        winner_id = max(0, min(winner_id, len(candidates) - 1))

        # Parse scores
        scores = self._parse_scores(response.parsed)

        # Display scores on UI if available
        if self.ui:
            self.ui.show_score_card(
                view_name,
                clarity=scores.clarity,
                specificity=scores.specificity,
                actionability=scores.actionability,
                creativity=scores.creativity,
                robustness=scores.robustness,
                conciseness=scores.conciseness,
            )

        return (
            candidates[winner_id],
            response.parsed.get("reason", ""),
            scores,
        )

    def final_judge(
        self,
        winners: dict[str, str]
    ) -> tuple[str, str, str, ScoreBreakdown]:
        """
        Stage 2: Judge the three view winners.
        
        Args:
            winners: {view_name: winning_prompt}
            
        Returns:
            (final_winner, winning_view, reason, scores)
        """

        # Extract views and prompts (maintain order: optimistic, critical, creative)
        views = list(winners.keys())
        prompts = list(winners.values())

        # Format for judge
        formatted = "\n\n".join(
            f"ID {i}:\n[{views[i].upper()}]\n{prompts[i]}"
            for i in range(len(prompts))
        )

        user_msg = Message(
            role="user",
            content=formatted,
        )

        # Send to judge for final evaluation
        response = self.send_for_json(
            [user_msg],
            system=self.FINAL_SYSTEM,
            retries=MAX_RETRIES_JSON,
        )

        # Extract winner ID and validate
        winner_id = response.parsed.get("winner_id", 0)

        if not isinstance(winner_id, int):
            winner_id = 0

        winner_id = max(0, min(winner_id, len(prompts) - 1))

        # Parse scores
        scores = self._parse_scores(response.parsed)

        return (
            prompts[winner_id],
            views[winner_id],
            response.parsed.get("reason", ""),
            scores,
        )

    def judge(
        self,
        view_candidates: dict[str, list[str]]
    ) -> dict:
        """
        Full two-stage judging pipeline.
        
        Args:
            view_candidates: {view_name: [candidate1, candidate2]}
            
        Returns:
            Complete judging results dict
        """

        intra_results = {}
        view_winners = {}
        view_scores = {}

        if self.ui:
            print("\n⏳ Parallel intra-view judging...")
        else:
            print("\n⏳ Parallel intra-view judging...")

        # Stage 1: Evaluate all views in parallel
        with ThreadPoolExecutor(
            max_workers=len(view_candidates)
        ) as pool:

            # Submit all tasks at once
            future_to_view = {
                pool.submit(
                    self.evaluate_view,
                    view,
                    candidates
                ): view
                for view, candidates in view_candidates.items()
            }

            # Collect results as they finish
            for future in as_completed(future_to_view):

                view = future_to_view[future]
                winner, reason, scores = future.result()

                intra_results[view] = {
                    "winner": winner,
                    "reason": reason,
                    "scores": scores.as_dict(),
                }

                view_winners[view] = winner
                view_scores[view] = scores

                # Update UI
                if self.ui:
                    self.ui.show_intra_view_progress(view, is_done=True)

        # Stage 2: Final cross-view judgment
        if self.ui:
            print("\n⏳ Cross-view judging...")

        final_winner, final_view, final_reason, final_scores = (
            self.final_judge(view_winners)
        )

        return {
            "intra_results": intra_results,
            "final_winner": final_winner,
            "final_view": final_view,
            "final_reason": final_reason,
            "final_scores": final_scores.as_dict(),
            "final_score_average": final_scores.average(),
        }


# ==============================================================================
# CACHING SYSTEM (In-Memory)
# ==============================================================================

_CACHE = {}  # Simple dict cache: hash(intent) → result


def _cache_lookup(intent: str):
    """
    Check if we've seen this intent before.
    
    Returns cached result if found, None otherwise.
    """
    return _CACHE.get(_intent_hash(intent))


def _cache_store(intent: str, result: dict):
    """Store result for future identical intents."""
    _CACHE[_intent_hash(intent)] = result


# ==============================================================================
# SUPABASE LOGGING (Optional Analytics)
# ==============================================================================

def _get_supabase_client():
    """
    Get Supabase client if credentials are configured.
    
    Returns None if Supabase is not available or not configured.
    
    Requires:
    - SUPABASE_AVAILABLE = True (supabase-py installed)
    - SUPABASE_URL environment variable
    - SUPABASE_KEY environment variable
    """
    if not SUPABASE_AVAILABLE:
        return None

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        return None

    return create_client(url, key)


def _log_run_to_supabase(
    intent: str,
    result: dict,
    context: RunContext,
    generation_time: float,
    judging_time: float,
    total_time: float,
):
    """
    Log one complete pipeline run to Supabase database.
    
    Silently fails if Supabase is not configured (doesn't break pipeline).
    
    Saves:
    - Intent hash (for cache deduplication)
    - Final winner prompt and reasoning
    - All 6 quality scores
    - Token counts and costs (USD + INR)
    - Timing for each phase
    - Timestamp
    
    This data enables:
    - Cost analysis dashboard
    - Quality trend analysis
    - Usage patterns
    - A/B testing results
    """

    supabase = _get_supabase_client()

    if not supabase:
        print("⚠️  Supabase not configured, skipping run logging")
        return

    try:

        supabase.table("pipeline_runs").insert({
            "intent_hash": _intent_hash(intent),
            "intent_length": len(intent),
            "final_winner": result.get("final_winner", ""),
            "final_view": result.get("final_view", ""),
            "final_reason": result.get("final_reason", ""),
            "final_scores": json.dumps(result.get("final_scores", {})),
            "input_tokens": context.input_tokens,
            "output_tokens": context.output_tokens,
            "total_tokens": context.input_tokens + context.output_tokens,
            "cost_usd": round(context.cost_usd, 6),
            "cost_inr": round(context.cost_inr, 2),
            "generation_time_sec": round(generation_time, 2),
            "judging_time_sec": round(judging_time, 2),
            "total_time_sec": round(total_time, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }).execute()

        print("✓ Run logged to Supabase")

    except Exception as e:
        print(f"⚠️  Supabase logging failed: {e}")


# ==============================================================================
# PIPELINE ORCHESTRATION HELPERS
# ==============================================================================

def _get_user_intent() -> str:
    """
    Prompt user for their intent.
    
    Displays fancy header and validates input is not empty.
    """

    ui = ScoreboardUI()
    ui.header("🚀 SINTARA V3 — Prompt Intelligence Engine")

    intent = input(f"{Colors.CYAN}Enter your intent:{Colors.RESET}\n> ").strip()

    if not intent:
        raise ValueError("Intent cannot be empty.")

    return intent


def _run_generator(
    gen: Generator,
    intent: str,
    ui: Optional[ScoreboardUI] = None,
):
    """
    Run one generator and track timing/cost.
    
    This function is designed to run in parallel via ThreadPoolExecutor.
    Each call is independent and thread-safe.
    
    Args:
        gen: Generator instance
        intent: User's intent
        ui: UI for display (optional)
        
    Returns:
        (generator_name, ViewPrompts)
    """

    if ui:
        ui.show_generation_progress(gen.name, done=False)
    else:
        print(f"⏳ [{gen.name}] generating...")

    # Record starting point for delta calculation
    start_cost = _current_context.cost_usd
    start_time = time.time()

    # Call generator (expensive operation)
    vp = gen.generate(intent)

    # Calculate delta
    duration = time.time() - start_time
    call_cost = _current_context.cost_usd - start_cost

    if ui:
        ui.show_generation_progress(
            gen.name,
            done=True,
            cost=call_cost,
            duration=duration,
            tokens_in=_current_context.input_tokens,
            tokens_out=_current_context.output_tokens,
        )

    return gen.name, vp


def _merge_view_candidates(results):
    """
    Reorganize generator outputs by view.
    
    INPUT (from generators):
    [
        ("GPT-5.1", ViewPrompts(opt="...", crit="...", creat="...")),
        ("Gemini", ViewPrompts(opt="...", crit="...", creat="...")),
    ]
    
    OUTPUT (for judge):
    {
        "optimistic": ["GPT's opt", "Gemini's opt"],
        "critical": ["GPT's crit", "Gemini's crit"],
        "creative": ["GPT's creat", "Gemini's creat"],
    }
    
    Judge needs candidates grouped by view, not by generator.
    """

    merged = {
        "optimistic": [],
        "critical": [],
        "creative": [],
    }

    for _, vp in results:

        for view, text in vp.as_dict().items():

            if text:
                merged[view].append(text)

    return merged


# ==============================================================================
# MAIN PIPELINE FUNCTION
# ==============================================================================

def run_full_pipeline(
    intent: str = None,
    use_diamond_mode: bool = False,
):
    """
    Complete prompt generation and evaluation pipeline.
    
    HIGH-LEVEL FLOW:
    1. Get user intent (ask or use provided)
    2. Check cache (skip if found)
    3. Run generators in parallel (GPT-5.1, Gemini, optional Claude)
    4. Merge results by view
    5. Stage 1: Parallel intra-view judging (Judge picks best per view)
    6. Stage 2: Cross-view judging (Judge picks overall best)
    7. Log to Supabase (for analytics)
    8. Cache result
    9. Display final results
    
    Args:
        intent: User's intent (if None, prompt for input)
        use_diamond_mode: If True, use 3 generators (add Claude Sonnet)
        
    Returns:
        Complete result dict with winner, scores, metrics
        
    Timing:
    - Generation phase: ~2-3 seconds (parallel)
    - Judging phase: ~4-5 seconds (parallel intra + sequential cross)
    - Total: ~7-8 seconds
    """

    # Initialize UI
    ui = ScoreboardUI()

    total_start = time.time()

    # Reset global context for this run
    global _current_context
    _current_context = RunContext()

    # ================================================
    # BUILD GENERATORS
    # ================================================

    generators = [
        Generator(
            OpenAIBackend(),
            name="GPT-5.1"
        ),
        Generator(
            GeminiBackend(),
            name="Gemini-3-Flash"
        ),
    ]

    # Optional: Add Claude Sonnet for Diamond tier (3 generators)
    if use_diamond_mode:
        generators.append(
            Generator(
                ClaudeBackend(model="claude-sonnet-4-6"),
                name="Claude-Sonnet"
            )
        )

    # ================================================
    # BUILD JUDGE
    # ================================================

    judge = Judge(
        model_backend=ClaudeBackend(
            model="claude-haiku-4-5"
        ),
        ui=ui,
    )

    # ================================================
    # GET INTENT
    # ================================================

    if intent is None:
        intent = _get_user_intent()

    # ================================================
    # CHECK CACHE
    # ================================================

    cached = _cache_lookup(intent)

    if cached:
        print(f"\n{Colors.BRIGHT_GREEN}⚡ Cache hit! Returning cached result...{Colors.RESET}\n")
        return cached

    # ================================================
    # PHASE 1: GENERATE (PARALLEL)
    # ================================================

    ui.header(f"🚀 Generating Prompts ({len(generators)} generators)")

    generation_start = time.time()

    results = []

    # Run all generators in parallel
    with ThreadPoolExecutor(max_workers=len(generators)) as pool:

        # Submit all tasks
        futures = {
            pool.submit(_run_generator, gen, intent, ui): gen.name
            for gen in generators
        }

        # Collect results as they finish (not in order)
        for future in as_completed(futures):
            results.append(future.result())

    generation_time = time.time() - generation_start

    # Sort results to match generator list order (for consistency)
    order = {gen.name: i for i, gen in enumerate(generators)}
    results.sort(key=lambda x: order.get(x[0], 999))

    # ================================================
    # MERGE VIEWS
    # ================================================

    view_candidates = _merge_view_candidates(results)

    print(
        f"\n{Colors.BRIGHT_GREEN}✓{Colors.RESET} Merged candidates: "
        f"{', '.join(f'{k}={len(v)}' for k, v in view_candidates.items())}"
    )

    # ================================================
    # PHASE 2 & 3: JUDGE (TWO-STAGE)
    # ================================================

    judging_start = time.time()
    judged = judge.judge(view_candidates)
    judging_time = time.time() - judging_start

    total_time = time.time() - total_start

    # ================================================
    # BUILD METRICS
    # ================================================

    judged["metrics"] = {
        "generation_time_sec": round(generation_time, 2),
        "judging_time_sec": round(judging_time, 2),
        "total_time_sec": round(total_time, 2),
        "input_tokens": _current_context.input_tokens,
        "output_tokens": _current_context.output_tokens,
        "total_tokens": (
            _current_context.input_tokens
            + _current_context.output_tokens
        ),
        "cost_usd": round(_current_context.cost_usd, 6),
        "cost_inr": round(_current_context.cost_inr, 2),
        "num_generators": len(generators),
        "mode": "diamond" if use_diamond_mode else "silver",
    }

    # ================================================
    # LOG TO SUPABASE
    # ================================================

    _log_run_to_supabase(
        intent,
        judged,
        _current_context,
        generation_time,
        judging_time,
        total_time,
    )

    # ================================================
    # CACHE RESULT
    # ================================================

    _cache_store(intent, judged)

    # ================================================
    # DISPLAY RESULTS
    # ================================================

    ui.show_final_results(judged, judged["metrics"])

    return judged


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    import sys

    # Check for --diamond flag in command line
    use_diamond = "--diamond" in sys.argv

    # Run pipeline
    result = run_full_pipeline(use_diamond_mode=use_diamond)

    print(f"\n{Colors.BRIGHT_GREEN}✓ Pipeline completed successfully{Colors.RESET}\n")
