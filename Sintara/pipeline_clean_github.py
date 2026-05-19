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

USD_TO_INR = 95.7
MAX_RETRIES_JSON = 3
MAX_RETRIES_API = 2

_COST_TABLE = {
    "gpt-5.1": (15.0, 60.0),
    "gemini-3-flash": (0.075, 0.30),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}

_WORDS_PER_TOKEN = 0.75


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / _WORDS_PER_TOKEN))


def _cost_for_model(model: str, inp: int, out: int) -> float:
    inp_rate, out_rate = _COST_TABLE.get(model, (1.0, 5.0))
    return (inp * inp_rate / 1_000_000) + (out * out_rate / 1_000_000)


def _intent_hash(intent: str) -> str:
    return hashlib.sha256(intent.strip().lower().encode()).hexdigest()


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ModelResponse:
    raw: str
    parsed: dict = field(default_factory=dict)


@dataclass
class ViewPrompts:
    optimistic: str = ""
    critical: str = ""
    creative: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "optimistic": self.optimistic,
            "critical": self.critical,
            "creative": self.creative,
        }


@dataclass
class ScoreBreakdown:
    clarity: int = 0
    specificity: int = 0
    actionability: int = 0
    creativity: int = 0
    robustness: int = 0
    conciseness: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "clarity": self.clarity,
            "specificity": self.specificity,
            "actionability": self.actionability,
            "creativity": self.creativity,
            "robustness": self.robustness,
            "conciseness": self.conciseness,
        }

    def average(self) -> float:
        scores = list(self.as_dict().values())
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class RunContext:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def cost_inr(self) -> float:
        return self.cost_usd * USD_TO_INR

    def add_call(self, inp: int, out: int, cost: float):
        self.input_tokens += inp
        self.output_tokens += out
        self.cost_usd += cost


_current_context = RunContext()


class ModelBase(abc.ABC):

    def send(
        self,
        messages: list[Message],
        *,
        system: str = ""
    ) -> ModelResponse:
        raw = self._call_model(messages, system=system)
        return ModelResponse(raw=raw)

    def send_for_json(
        self,
        messages: list[Message],
        *,
        system: str = "",
        retries: int = MAX_RETRIES_JSON,
    ) -> ModelResponse:

        json_instruction = (
            "Respond with valid JSON only. "
            "No markdown. No explanations. No preamble."
        )

        json_system = (
            f"{system}\n\n{json_instruction}"
            if system else json_instruction
        )

        last_raw = ""

        for attempt in range(retries + 1):
            raw = self._call_model(messages, system=json_system)
            last_raw = raw
            cleaned = self._strip_fences(raw)

            try:
                parsed = json.loads(cleaned)
                return ModelResponse(raw=raw, parsed=parsed)
            except json.JSONDecodeError:
                if attempt < retries:
                    continue

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
        pass

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()


class ClaudeBackend(ModelBase):

    DEFAULT_MODEL = "claude-haiku-4-5"
    DEFAULT_MAX_TOKENS = 2048

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
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

        api_msgs = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=api_msgs,
        )

        if system:
            kwargs["system"] = system

        start = time.time()
        response = self._client.messages.create(**kwargs)
        duration = time.time() - start

        text = response.content[0].text

        inp = getattr(
            response.usage,
            "input_tokens",
            _estimate_tokens(system + text)
        )

        out = getattr(
            response.usage,
            "output_tokens",
            _estimate_tokens(text)
        )

        cost = _cost_for_model(self.model, inp, out)
        _current_context.add_call(inp, out, cost)

        return text


class OpenAIBackend(ModelBase):

    DEFAULT_MODEL = "gpt-5.1"
    DEFAULT_MAX_TOKENS = 2048

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
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

        start = time.time()
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=api_msgs,
        )
        duration = time.time() - start

        text = response.choices[0].message.content or ""

        if response.usage:
            inp = response.usage.prompt_tokens
            out = response.usage.completion_tokens
        else:
            inp = _estimate_tokens(system + text)
            out = _estimate_tokens(text)

        cost = _cost_for_model(self.model, inp, out)
        _current_context.add_call(inp, out, cost)

        return text


class GeminiBackend(ModelBase):

    DEFAULT_MODEL = "gemini-3-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
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

        parts = []
        if system:
            parts.append(system)

        parts += [m.content for m in messages]

        start = time.time()
        response = self._client.generate_content(
            "\n\n".join(parts)
        )
        duration = time.time() - start

        text = response.text

        if hasattr(response, 'usage_metadata'):
            inp = response.usage_metadata.input_token_count
            out = response.usage_metadata.output_token_count
        else:
            inp = _estimate_tokens(system + text)
            out = _estimate_tokens(text)

        cost = _cost_for_model(self.model, inp, out)
        _current_context.add_call(inp, out, cost)

        return text


class Generator(ModelBase):

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
        self._backend = model_backend
        self.name = name or type(model_backend).__name__
        self.max_tokens = max_tokens

    def _call_model(
        self,
        messages: list[Message],
        *,
        system: str = ""
    ) -> str:
        return self._backend._call_model(messages, system=system)

    def generate(self, intent: str) -> ViewPrompts:
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


class Judge(ModelBase):

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
        self._backend = model_backend or ClaudeBackend()
        self.ui = ui

    def _call_model(
        self,
        messages: list[Message],
        *,
        system: str = ""
    ) -> str:
        return self._backend._call_model(messages, system=system)

    def _parse_scores(self, parsed: dict) -> ScoreBreakdown:
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

        if len(candidates) == 1:
            scores = ScoreBreakdown()
            return candidates[0], "Only one candidate available.", scores

        numbered = "\n\n".join(
            f"ID {i}:\n{candidate}"
            for i, candidate in enumerate(candidates)
        )

        user_msg = Message(
            role="user",
            content=f"VIEW: {view_name}\n\n{numbered}"
        )

        response = self.send_for_json(
            [user_msg],
            system=self.INTRA_SYSTEM,
            retries=MAX_RETRIES_JSON,
        )

        winner_id = response.parsed.get("winner_id", 0)

        if not isinstance(winner_id, int):
            winner_id = 0

        winner_id = max(0, min(winner_id, len(candidates) - 1))

        scores = self._parse_scores(response.parsed)

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

        views = list(winners.keys())
        prompts = list(winners.values())

        formatted = "\n\n".join(
            f"ID {i}:\n[{views[i].upper()}]\n{prompts[i]}"
            for i in range(len(prompts))
        )

        user_msg = Message(
            role="user",
            content=formatted,
        )

        response = self.send_for_json(
            [user_msg],
            system=self.FINAL_SYSTEM,
            retries=MAX_RETRIES_JSON,
        )

        winner_id = response.parsed.get("winner_id", 0)

        if not isinstance(winner_id, int):
            winner_id = 0

        winner_id = max(0, min(winner_id, len(prompts) - 1))

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

        intra_results = {}
        view_winners = {}
        view_scores = {}

        if self.ui:
            print("\n⏳ Parallel intra-view judging...")
        else:
            print("\n⏳ Parallel intra-view judging...")

        with ThreadPoolExecutor(
            max_workers=len(view_candidates)
        ) as pool:

            future_to_view = {
                pool.submit(
                    self.evaluate_view,
                    view,
                    candidates
                ): view
                for view, candidates in view_candidates.items()
            }

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

                if self.ui:
                    self.ui.show_intra_view_progress(view, is_done=True)

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


_CACHE = {}


def _cache_lookup(intent: str):
    return _CACHE.get(_intent_hash(intent))


def _cache_store(intent: str, result: dict):
    _CACHE[_intent_hash(intent)] = result


def _get_supabase_client():
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


def _get_user_intent() -> str:

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

    if ui:
        ui.show_generation_progress(gen.name, done=False)
    else:
        print(f"⏳ [{gen.name}] generating...")

    start_cost = _current_context.cost_usd
    start_time = time.time()

    vp = gen.generate(intent)

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


def run_full_pipeline(
    intent: str = None,
    use_diamond_mode: bool = False,
):

    ui = ScoreboardUI()

    total_start = time.time()

    global _current_context
    _current_context = RunContext()

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

    if use_diamond_mode:
        generators.append(
            Generator(
                ClaudeBackend(model="claude-sonnet-4-6"),
                name="Claude-Sonnet"
            )
        )

    judge = Judge(
        model_backend=ClaudeBackend(
            model="claude-haiku-4-5"
        ),
        ui=ui,
    )

    if intent is None:
        intent = _get_user_intent()

    cached = _cache_lookup(intent)

    if cached:
        print(f"\n{Colors.BRIGHT_GREEN}⚡ Cache hit! Returning cached result...{Colors.RESET}\n")
        return cached

    ui.header(f"🚀 Generating Prompts ({len(generators)} generators)")

    generation_start = time.time()

    results = []

    with ThreadPoolExecutor(max_workers=len(generators)) as pool:

        futures = {
            pool.submit(_run_generator, gen, intent, ui): gen.name
            for gen in generators
        }

        for future in as_completed(futures):
            results.append(future.result())

    generation_time = time.time() - generation_start

    order = {gen.name: i for i, gen in enumerate(generators)}
    results.sort(key=lambda x: order.get(x[0], 999))

    view_candidates = _merge_view_candidates(results)

    print(
        f"\n{Colors.BRIGHT_GREEN}✓{Colors.RESET} Merged candidates: "
        f"{', '.join(f'{k}={len(v)}' for k, v in view_candidates.items())}"
    )

    judging_start = time.time()
    judged = judge.judge(view_candidates)
    judging_time = time.time() - judging_start

    total_time = time.time() - total_start

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

    _log_run_to_supabase(
        intent,
        judged,
        _current_context,
        generation_time,
        judging_time,
        total_time,
    )

    _cache_store(intent, judged)

    ui.show_final_results(judged, judged["metrics"])

    return judged


if __name__ == "__main__":

    import sys

    use_diamond = "--diamond" in sys.argv

    result = run_full_pipeline(use_diamond_mode=use_diamond)

    print(f"\n{Colors.BRIGHT_GREEN}✓ Pipeline completed successfully{Colors.RESET}\n")
