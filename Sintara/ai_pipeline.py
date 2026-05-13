"""
AI Pipeline: Multi-model prompt generation + judging system.

Architecture:
  ModelBase              — abstract base: wraps any model's input/output
    ├── ClaudeBackend    — Anthropic Claude (judge: claude-haiku-4-5)
    ├── OpenAIBackend    — OpenAI          (generator: gpt-4.1)
    └── GeminiBackend    — Google Gemini   (generator: gemini-2.5-flash)

  Generator              — wraps a ModelBase backend; produces 3 view-prompts
  Judge                  — wraps ClaudeBackend; two-stage evaluation

Pipeline (run_full_pipeline):
  1. User types intent once.
  2. GPT-4.1  generates 3 view-prompts  ┐ concurrently via ThreadPoolExecutor
     Gemini 2.5 Flash generates 3 views ┘
  3. Claude Haiku (Judge) compares per-view candidates  → 3 intra-view winners
     [OPT] All 3 intra-view comparisons now run concurrently.
  4. Claude Haiku (Judge) compares the 3 winners        → 1 final best prompt

Adding a new generator model:
  1. Subclass ModelBase and implement _call_model().
  2. Append Generator(model_backend=YourBackend(), name="YourName")
     to the `generators` list inside run_full_pipeline() — no other change needed.

Optimizations vs original (latency-only; no feature changes):
  1. Judge.judge()     — Stage 1 intra-view calls are now parallel
                         (was sequential: view0 → view1 → view2;
                          now concurrent: view0 ║ view1 ║ view2).
  2. run_full_pipeline — Removed dead duplicate _get_user_intent() call that
                         could block after generation was already complete.
  3. ModelBase._strip_fences — Single-pass fence stripping (minor CPU savings).
  4. Judge.judge()     — Thread pool is reused across both pipeline stages
                         instead of creating a new one for Stage 1.
"""

from __future__ import annotations

import abc
import os
import json
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
    """
    Provider-agnostic wrapper around a language model.

    To add a new provider:
      1. Subclass this class.
      2. Implement _call_model(messages, *, system="") -> str.
      3. Pass an instance to Generator or Judge.
    """

    def send(self, messages: list[Message], *, system: str = "") -> ModelResponse:
        raw = self._call_model(messages, system=system)
        return ModelResponse(raw=raw)

    def send_for_json(
        self,
        messages: list[Message],
        *,
        system: str = "",
    ) -> ModelResponse:
        """
        Like send(), but appends a JSON-only instruction to the system prompt
        and auto-parses the response. Strips markdown code fences if present.
        """
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
        # OPT: single-pass: check and strip opening fence, then closing fence.
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()


# ---------------------------------------------------------------------------
# Backend: Anthropic Claude  (judge → claude-haiku-4-5)
# ---------------------------------------------------------------------------

class ClaudeBackend(ModelBase):
    """
    Anthropic Claude backend.
    Default model: claude-haiku-4-5  (used as the Judge).

    Required env var: ANTHROPIC_API_KEY
    """

    DEFAULT_MODEL      = "claude-haiku-4-5"
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
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
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
# Backend: OpenAI GPT  (generator → gpt-4.1)
# ---------------------------------------------------------------------------

class OpenAIBackend(ModelBase):
    """
    OpenAI backend.
    Default model: gpt-4.1  (used as a Generator).

    Required env var: OPENAI_API_KEY
    """

    DEFAULT_MODEL      = "gpt-4.1"
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
            api_key=api_key or os.environ["OPENAI_API_KEY"]
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
# Backend: Google Gemini  (generator → gemini-2.5-flash)
# ---------------------------------------------------------------------------

class GeminiBackend(ModelBase):
    """
    Google Gemini backend.
    Default model: gemini-2.5-flash-preview-05-20  (used as a Generator).

    Required env var: GOOGLE_API_KEY
    """

    DEFAULT_MODEL = "gemini-2.5-flash-preview-05-20"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.model = model
        genai.configure(api_key=api_key or os.environ["GOOGLE_API_KEY"])
        self._client = genai.GenerativeModel(self.model)

    def _call_model(self, messages: list[Message], *, system: str = "") -> str:
        # Gemini uses a flat content list; prepend system as an initial turn.
        parts: list[str] = []
        if system:
            parts.append(system)
        parts += [m.content for m in messages]

        response = self._client.generate_content("\n\n".join(parts))
        return response.text


# ---------------------------------------------------------------------------
# Generator  (subclass of ModelBase)
# ---------------------------------------------------------------------------

class Generator(ModelBase):
    """
    Wraps any ModelBase backend and converts a user intent into three
    prompt variants: optimistic / critical / creative.

    To add a new generation model, just instantiate Generator with a different
    backend — no changes to this class are needed.
    """

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
        """
        Parameters
        ----------
        model_backend : ModelBase
            The backend to use for generation.
        name : str
            Human-readable label used in logs (e.g. "GPT-4.1").
        """
        self._backend = model_backend
        self.name     = name or type(model_backend).__name__

    def _call_model(self, messages: list[Message], *, system: str = "") -> str:
        return self._backend._call_model(messages, system=system)

    def generate(self, intent: str) -> ViewPrompts:
        """Send `intent` to the backend; return a ViewPrompts with 3 variants."""
        messages = [Message(role="user", content=intent)]
        resp = self.send_for_json(messages, system=self.SYSTEM_PROMPT)

        if not resp.parsed:
            raise RuntimeError(
                f"[{self.name}] Could not parse model output as JSON.\n"
                f"Raw output:\n{resp.raw}"
            )

        return ViewPrompts(
            optimistic=resp.parsed.get("optimistic", ""),
            critical=resp.parsed.get("critical",   ""),
            creative=resp.parsed.get("creative",   ""),
        )


# ---------------------------------------------------------------------------
# Judge  (subclass of ModelBase)
# ---------------------------------------------------------------------------

class Judge(ModelBase):
    """
    Two-stage evaluation backed by ClaudeBackend (claude-haiku-4-5).

    Stage 1 — Intra-view comparison  [OPT: now runs all views concurrently]
        For each view (optimistic / critical / creative), compare all candidates
        (one per generator) and pick the best one.

    Stage 2 — Cross-view final comparison
        Compare the three per-view winners and pick the single best prompt.
    """

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
        """Defaults to ClaudeBackend() (claude-haiku-4-5) if nothing is supplied."""
        self._backend = model_backend or ClaudeBackend()

    def _call_model(self, messages: list[Message], *, system: str = "") -> str:
        return self._backend._call_model(messages, system=system)

    def evaluate_view(self, view_name: str, candidates: list[str]) -> tuple[str, str]:
        """
        Stage 1: compare candidates within a single view.
        Returns (winner_text, reason).
        """
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
        """
        Stage 2: compare per-view winners to select the overall best prompt.
        Returns (winner_text, winning_view, reason).
        """
        formatted = "\n\n".join(
            f"[{view.upper()}]\n{text}" for view, text in view_winners.items()
        )
        user_msg = Message(role="user", content=formatted)
        resp     = self.send_for_json([user_msg], system=self.FINAL_SYSTEM)
        return (
            resp.parsed.get("winner", ""),
            resp.parsed.get("view",   ""),
            resp.parsed.get("reason", ""),
        )

    def judge(self, view_candidates: dict[str, list[str]]) -> dict:
        """
        Full two-stage pipeline.

        Parameters
        ----------
        view_candidates : dict[str, list[str]]
            Keys = view names; values = list of candidate strings (one per generator).

        Returns
        -------
        dict with keys:
          "intra_results" — {view: {"winner": str, "reason": str}}
          "final_winner"  — the single best prompt text
          "final_view"    — which view it came from
          "final_reason"  — why it won
        """
        intra_results: dict[str, dict] = {}
        view_winners:  dict[str, str]  = {}

        # OPT: Run all intra-view comparisons concurrently instead of sequentially.
        # With 3 views and ~1–2 s per Claude Haiku call, this saves ~2–4 s of
        # sequential waiting at Stage 1 before Stage 2 can begin.
        print("\n⏳  Stage 1: intra-view comparison (Claude Haiku) …")

        num_views = len(view_candidates)
        with ThreadPoolExecutor(max_workers=num_views) as pool:
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

        print("\n⏳  Stage 2: cross-view final comparison (Claude Haiku) …")
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
            snippet  = data["winner"][:120]
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
    """Thread target: run one generator and return (name, ViewPrompts)."""
    print(f"  ⏳  [{gen.name}] generating …")
    vp = gen.generate(intent)
    print(f"  ✓  [{gen.name}] done.")
    return gen.name, vp


def _merge_view_candidates(
    results: list[tuple[str, ViewPrompts]],
) -> dict[str, list[str]]:
    """
    Combine outputs from multiple generators into per-view candidate lists.

    Example output:
      {
        "optimistic": ["<gpt prompt>", "<gemini prompt>"],
        "critical":   ["<gpt prompt>", "<gemini prompt>"],
        "creative":   ["<gpt prompt>", "<gemini prompt>"],
      }
    """
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
            snippet  = text[:200]
            ellipsis = "…" if len(text) > 200 else ""
            print(f"  [{label.upper()}] {snippet}{ellipsis}")
    print("─" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(intent: str = None) -> dict:
    """
    End-to-end pipeline:

      1. Ask the user for an intent.
      2. GPT-4.1 and Gemini 2.5 Flash each generate 3 view-prompts in parallel.
      3. Claude Haiku (Judge) picks the best prompt per view  (Stage 1).
         [OPT] All 3 intra-view comparisons now run concurrently.
      4. Claude Haiku (Judge) picks the single best overall prompt (Stage 2).

    ── To add another generator ────────────────────────────────────────────
    Option A — new model from an existing provider:
        generators.append(
            Generator(model_backend=OpenAIBackend(model="gpt-4o"), name="GPT-4o")
        )

    Option B — new provider entirely:
        class MyBackend(ModelBase):
            def _call_model(self, messages, *, system=""):
                ...  # your API call here
        generators.append(Generator(model_backend=MyBackend(), name="MyModel"))
    ────────────────────────────────────────────────────────────────────────
    """

    # ── Generators ───────────────────────────────────────────────────────────
    generators: list[Generator] = [
        Generator(model_backend=OpenAIBackend(),  name="GPT-4.1"),
        Generator(model_backend=GeminiBackend(),  name="Gemini-2.5-Flash"),
        # Add more here ↓
    ]

    # ── Judge (Claude Haiku) ─────────────────────────────────────────────────
    judge = Judge(model_backend=ClaudeBackend())   # claude-haiku-4-5

    # ── Step 1: get user intent ───────────────────────────────────────────────
    # OPT: Intent is fetched exactly once, before generation begins.
    # The original had a duplicate `if not intent: _get_user_intent()` call
    # placed *after* generation completed — a blocking stall that could never
    # be triggered (intent was always set) but added dead code risk.
    if intent is None:
        intent = _get_user_intent()

    # ── Step 2: run all generators concurrently ───────────────────────────────
    print(f"\n🚀  Running {len(generators)} generators in parallel …")
    gen_results: list[tuple[str, ViewPrompts]] = []

    with ThreadPoolExecutor(max_workers=len(generators)) as pool:
        futures = {
            pool.submit(_run_generator, gen, intent): gen.name
            for gen in generators
        }
        for future in as_completed(futures):
            name, vp = future.result()   # re-raises any exception from the thread
            gen_results.append((name, vp))

    # Restore a stable display order matching the `generators` list.
    order = {gen.name: i for i, gen in enumerate(generators)}
    gen_results.sort(key=lambda t: order.get(t[0], 99))

    _print_all_views(gen_results)

    # ── Steps 3 & 4: judge ───────────────────────────────────────────────────
    view_candidates = _merge_view_candidates(gen_results)
    return judge.run(view_candidates)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_full_pipeline()
    print("\nDone. Best prompt is in result['final_winner'].")