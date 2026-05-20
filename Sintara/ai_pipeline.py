"""
AI Pipeline: Multi-model prompt generation + judging system.

Architecture:
  ModelBase              — abstract base: wraps any model's input/output
    ├── ClaudeBackend    — Anthropic Claude (judge: claude-sonnet-4-6)
    ├── OpenAIBackend    — OpenAI         (generator: gpt-5.1)
    └── GeminiBackend    — Google Gemini   (generator: gemini-3.5-flash-preview)

  Generator              — wraps a ModelBase backend; produces 3 view-prompts
  Judge                  — wraps ClaudeBackend; two-stage evaluation
"""

from __future__ import annotations

import abc
import os
import json
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import anthropic                       # pip install anthropic
import openai                          # pip install openai
import google.generativeai as genai    # pip install google-generativeai


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str      # "user" | "assistant"   (system is passed separately)
    content: str


@dataclass
class ModelResponse:
    raw: str
    parsed: dict = field(default_factory=dict)


@dataclass
class ViewPrompts:
    """Three prompt variants produced by one Generator for a single intent."""
    optimistic: str = ""
    critical:   str = ""
    creative:   str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "optimistic": self.optimistic,
            "critical":   self.critical,
            "creative":   self.creative,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ModelBase(abc.ABC):
    def send(self, messages: list[Message], *, system: str = "") -> ModelResponse:
        raw = self._call_model(messages, system=system)
        return ModelResponse(raw=raw)

    def send_for_json(
        self,
        messages: list[Message],
        *,
        system: str = "",
    ) -> ModelResponse:
        json_instruction = (
            "Respond with valid JSON only. "
            "Do not include any prose, explanation, or markdown code fences."
        )
        json_system = f"{system}\n\n{json_instruction}" if system else json_instruction
        raw = self._call_model(messages, system=json_system)
        cleaned = self._strip_fences(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = {}
        return ModelResponse(raw=raw, parsed=parsed)

    @abc.abstractmethod
    def _call_model(self, messages: list[Message], *, system: str = "") -> str:
        """Make the API call; return the model's reply as a plain string."""

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        # 1. Check for markdown code blocks anywhere in the response
        match = re.search(r'```(?:json)?\s*(.*?)\s*
```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 2. Fallback: Find the first { and last } in case the model is chatty
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
            
        return text.strip()


# ---------------------------------------------------------------------------
# Backend: Anthropic Claude  (judge → claude-sonnet-4-6)
# ---------------------------------------------------------------------------

class ClaudeBackend(ModelBase):
    DEFAULT_MODEL      = "claude-sonnet-4-6"
    DEFAULT_MAX_TOKENS = 2048

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model      = model
        self.max_tokens = max_tokens
        self._client    = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

    def _call_model(self, messages: list[Message], *, system: str = "") -> str:
        api_msgs = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=api_msgs,
        )
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text


# ---------------------------------------------------------------------------
# Backend: OpenAI GPT  (generator → gpt-5.1)
# ---------------------------------------------------------------------------

class OpenAIBackend(ModelBase):
    DEFAULT_MODEL      = "gpt-5.1"
    DEFAULT_MAX_TOKENS = 2048

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model      = model
        self.max_tokens = max_tokens
        self._client    = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

    def _call_model(self, messages: list[Message], *, system: str = "") -> str:
        api_msgs: list[dict] = []
        if system:
            api_msgs.append({"role": "system", "content": system})
        api_msgs += [{"role": m.role, "content": m.content} for m in messages]

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=api_msgs,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Backend: Google Gemini  (generator → gemini-3.5-flash-preview)
# ---------------------------------------------------------------------------

class GeminiBackend(ModelBase):
    DEFAULT_MODEL = "gemini-3.5-flash-preview"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.model = model
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if key:
            genai.configure(api_key=key)

    def _call_model(self, messages: list[Message], *, system: str = "") -> str:
        client = genai.GenerativeModel(
            self.model, 
            system_instruction=system or None
        )
        parts = [m.content for m in messages]
        
        response = client.generate_content(parts)
        try:
            return response.text
        except ValueError:
            return "{}"


# ---------------------------------------------------------------------------
# Generator  (subclass of ModelBase)
# ---------------------------------------------------------------------------

class Generator(ModelBase):
    SYSTEM_PROMPT = textwrap.dedent("""\
        You are an expert prompt engineer.
        Given a user's intent, produce exactly THREE refined prompt variants.
        Each variant must be self-contained and ready to hand to an AI system.

        Return a JSON object with exactly these keys:
          "optimistic"  — a prompt that frames the intent positively and expansively
          "critical"    — a prompt that probes edge-cases, risks, and limitations
          "creative"    — a prompt that takes an unexpected or lateral angle

        No other keys. No prose outside the JSON object.
    """)

    def __init__(self, model_backend: ModelBase, name: str = "") -> None:
        self._backend = model_backend
        self.name     = name or type(model_backend).__name__

    def _call_model(self, messages: list[Message], *, system: str = "") -> str:
        return self._backend._call_model(messages, system=system)

    def generate(self, intent: str) -> ViewPrompts:
        messages = [Message(role="user", content=intent)]
        resp = self.send_for_json(messages, system=self.SYSTEM_PROMPT)

        if not resp.parsed:
            raise RuntimeError(f"[{self.name}] JSON parse failed.\nRaw:\n{resp.raw}")

        return ViewPrompts(
            optimistic=resp.parsed.get("optimistic", ""),
            critical=resp.parsed.get("critical",    ""),
            creative=resp.parsed.get("creative",    ""),
        )


# ---------------------------------------------------------------------------
# Judge  (subclass of ModelBase)
# ---------------------------------------------------------------------------

class Judge(ModelBase):
    INTRA_SYSTEM = textwrap.dedent("""\
        You are a rigorous prompt quality judge.
        You will receive several prompt candidates for the same VIEW.
        Select the single best candidate and explain your choice briefly.

        Return JSON with exactly:
          "winner" — the full text of the best prompt (verbatim, unchanged)
          "reason" — one or two sentences explaining why it wins

        No other keys. No prose outside the JSON object.
    """)

    FINAL_SYSTEM = textwrap.dedent("""\
        You are a rigorous prompt quality judge.
        You will receive three prompt candidates from different "views":
          optimistic, critical, and creative.
        Pick the single prompt most likely to produce a high-quality, useful AI
        response in real-world usage.

        Return JSON with exactly:
          "winner" — the full text of the best prompt (verbatim, unchanged)
          "view"   — which view it came from ("optimistic"|"critical"|"creative")
          "reason" — one or two sentences explaining your choice

        No other keys. No prose outside the JSON object.
    """)

    def __init__(self, model_backend: Optional[ModelBase] = None) -> None:
        self._backend = model_backend or ClaudeBackend()

    def _call_model(self, messages: list[Message], *, system: str = "") -> str:
        return self._backend._call_model(messages, system=system)

    def evaluate_view(self, view_name: str, candidates: list[str]) -> tuple[str, str]:
        if not candidates:
            return "", "No candidates were successfully generated for this view."
        
        if len(candidates) == 1:
            return candidates[0], "Only one candidate — selected by default."

        numbered = "\n\n".join(
            f"Candidate {i + 1}:\n{c}" for i, c in enumerate(candidates)
        )
        user_msg = Message(role="user", content=f"VIEW: {view_name}\n\n{numbered}")
        resp     = self.send_for_json([user_msg], system=self.INTRA_SYSTEM)
        winner   = resp.parsed.get("winner", candidates[0])
        reason   = resp.parsed.get("reason", "")
        return winner, reason

    def final_judge(self, view_winners: dict[str, str]) -> tuple[str, str, str]:
        valid_winners = {v: t for v, t in view_winners.items() if t}
        if not valid_winners:
            return "", "", "Pipeline failed: No prompts were generated."

        if len(valid_winners) == 1:
            view, text = list(valid_winners.items())[0]
            return text, view, "Only one view produced a valid prompt."

        formatted = "\n\n".join(
            f"[{view.upper()}]\n{text}" for view, text in valid_winners.items()
        )
        user_msg = Message(role="user", content=formatted)
        resp     = self.send_for_json([user_msg], system=self.FINAL_SYSTEM)
        
        return (
            resp.parsed.get("winner", ""),
            resp.parsed.get("view",   ""),
            resp.parsed.get("reason", ""),
        )

    def judge(self, view_candidates: dict[str, list[str]]) -> dict:
        intra_results: dict[str, dict] = {}
        view_winners:  dict[str, str]  = {}

        print("\n⏳  Stage 1: intra-view comparison (Claude Sonnet) …")
        num_views = len(view_candidates)
        
        with ThreadPoolExecutor(max_workers=max(1, num_views)) as pool:
            future_to_view = {
                pool.submit(self.evaluate_view, view_name, candidates): view_name
                for view_name, candidates in view_candidates.items()
            }
            for future in as_completed(future_to_view):
                view_name        = future_to_view[future]
                winner, reason   = future.result()
                intra_results[view_name] = {"winner": winner, "reason": reason}
                view_winners[view_name]  = winner
                print(f"  ✓ [{view_name}] winner selected.")

        print("\n⏳  Stage 2: cross-view final comparison (Claude Sonnet) …")
        final_winner, final_view, final_reason = self.final_judge(view_winners)

        return {
            "intra_results": intra_results,
            "final_winner":  final_winner,
            "final_view":    final_view,
            "final_reason":  final_reason,
        }

    def run(self, view_candidates: dict[str, list[str]]) -> dict:
        result = self.judge(view_candidates)
        self._print_result(result)
        return result

    @staticmethod
    def _print_result(result: dict) -> None:
        print("\n" + "=" * 60)
        print("  JUDGING RESULTS")
        print("=" * 60)

        print("\n── Intra-view winners ──")
        for view, data in result["intra_results"].items():
            snippet  = data["winner"][:120].replace('\n', ' ')
            ellipsis = "…" if len(data["winner"]) > 120 else ""
            print(f"\n  [{view.upper()}]")
            print(f"  Reason : {data['reason']}")
            print(f"  Winner : {snippet}{ellipsis}")

        print("\n── Final winner ──")
        print(f"  View   : {result['final_view'].upper()}")
        print(f"  Reason : {result['final_reason']}")
        print(f"\n  ★ BEST PROMPT:\n  {result['final_winner']}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_user_intent() -> str:
    print("\n" + "=" * 60)
    print("  AI PIPELINE — Enter your intent")
    print("=" * 60)
    intent = input("What do you want the AI to help with?\n> ").strip()
    if not intent:
        raise ValueError("Intent cannot be empty.")
    return intent


def _run_generator(gen: Generator, intent: str) -> tuple[str, ViewPrompts]:
    print(f"  ⏳  [{gen.name}] generating …")
    try:
        vp = gen.generate(intent)
        print(f"  ✓  [{gen.name}] done.")
        return gen.name, vp
    except Exception as e:
        print(f"  ❌  [{gen.name}] failed: {e}")
        return gen.name, ViewPrompts() 


def _merge_view_candidates(
    results: list[tuple[str, ViewPrompts]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {"optimistic": [], "critical": [], "creative": []}
    for _name, vp in results:
        for view, text in vp.as_dict().items():
            if text:
                merged[view].append(text)
    return merged


def _print_all_views(results: list[tuple[str, ViewPrompts]]) -> None:
    print("\n" + "─" * 60)
    print("Generated variants (all generators):")
    for name, vp in results:
        print(f"\n  ── {name} ──")
        for label, text in vp.as_dict().items():
            if not text:
                print(f"  [{label.upper()}] (Failed to generate)")
                continue
            snippet  = text[:200].replace('\n', ' ')
            ellipsis = "…" if len(text) > 200 else ""
            print(f"  [{label.upper()}] {snippet}{ellipsis}")
    print("─" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(intent: str = None) -> dict:
    generators: list[Generator] = [
        Generator(model_backend=OpenAIBackend(),  name="GPT-5.1"),
        Generator(model_backend=GeminiBackend(),  name="Gemini-3.5-Flash-Preview"),
    ]

    judge = Judge(model_backend=ClaudeBackend())   # claude-sonnet-4-6

    if intent is None:
        intent = _get_user_intent()

    print(f"\n🚀  Running {len(generators)} generators in parallel …")
    gen_results: list[tuple[str, ViewPrompts]] = []

    with ThreadPoolExecutor(max_workers=len(generators)) as pool:
        futures = {
            pool.submit(_run_generator, gen, intent): gen.name
            for gen in generators
        }
        for future in as_completed(futures):
            name, vp = future.result()   
            gen_results.append((name, vp))

    order = {gen.name: i for i, gen in enumerate(generators)}
    gen_results.sort(key=lambda t: order.get(t[0], 99))

    _print_all_views(gen_results)

    view_candidates = _merge_view_candidates(gen_results)
    return judge.run(view_candidates)


if __name__ == "__main__":
    result = run_full_pipeline()
    print("\nDone. Best prompt is in result['final_winner'].")
