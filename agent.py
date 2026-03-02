#!/usr/bin/env python3
import copy
import json
import sys
import time
from pathlib import Path
from typing import Optional

import requests

try:
    import toons
except ImportError as e:
    raise SystemExit(
        "ERROR: Install TOON support:\n"
        "  python3 -m pip install --user toons"
    ) from e


URL = "https://api.openai.com/v1/responses"


def load_pricing_models(script_dir: Optional[Path] = None) -> dict:
    """Load pricing models from pricing.json."""
    base = script_dir or Path(__file__).resolve().parent
    path = base / "pricing.json"

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("models", {}) or {}
    except Exception:
        print(f"[WARN] pricing.json not found, cost calculation disabled")
        return {}


def format_money(x: float) -> str:
    """Format float as USD."""
    return f"${x:.6f}"


def compute_cost(pricing: dict, model: str, in_tok, out_tok):
    """Compute cost from token usage and pricing."""
    p = pricing.get(model)
    if not p or in_tok is None or out_tok is None:
        return "unknown", "unknown", "unknown"
    try:
        in_cost = in_tok * float(p["input"]) / 1_000_000
        out_cost = out_tok * float(p["output"]) / 1_000_000
    except Exception:
        return "unknown", "unknown", "unknown"
    return (
        format_money(in_cost),
        format_money(out_cost),
        format_money(in_cost + out_cost),
    )


class ShortTermMemory:
    """Short-term memory: current dialogue session only (messages + summary)."""

    def __init__(self, history_limit: int, summary_chunk_size: int):
        """Initialize with empty messages and summary."""
        self.messages: list[dict] = []
        self.summary: str = ""
        self.history_limit: int = history_limit
        self.summary_chunk_size: int = summary_chunk_size

    def to_dict(self) -> dict:
        """Serialize to plain dict for TOON persistence."""
        return {
            "messages": self.messages,
            "summary": self.summary,
        }

    def from_dict(self, data: dict):
        """Restore from plain dict."""
        self.messages = data.get("messages", []) or []
        self.summary = data.get("summary", "") or ""


class Agent:
    """
    Minimal LLM agent with:
    - history
    - stage-based state
    - TOON persistence
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        history_limit: int,
        timeout: int,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        stop: list[str] | None = None,
        pricing: dict | None = None,
        print_json: bool = False,
        context_strategy: str = "window",
        context_summary: bool = False,
    ):
        """Initialize agent."""
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.history_limit = max(0, history_limit)
        self.timeout = timeout
        self.context_strategy = (context_strategy or "window").strip().lower()
        if self.context_strategy not in ("window", "facts", "branch"):
            raise ValueError(
                "Invalid context_strategy. Use: window | facts | branch"
            )
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.stop = stop or []
        self.print_json = print_json
        self.pricing = pricing or {}

        # Day 9: summary compression flag (config, not state)
        self.context_summary = context_summary

        # Day 11: short-term memory layer
        self.stm = ShortTermMemory(
            history_limit=self.history_limit,
            summary_chunk_size=10,
        )

        # Persistent state (working memory)
        self.stage = "IDLE"
        self.goal = ""
        self.plan = []
        self.actions = []
        self.notes = []
        self.facts: dict[str, str] = {}
        self.current_branch: str = "main"
        self.branches: dict[str, dict] = {"main": self._snapshot()}

    # ---------------- State management ----------------

    def _snapshot(self) -> dict:
        """Return a deep copy of current per-branch state (working + stm)."""
        return {
            "history": copy.deepcopy(self.stm.messages),
            "summary": self.stm.summary,
            "facts": copy.deepcopy(self.facts),
            "stage": self.stage,
            "goal": self.goal,
            "plan": copy.deepcopy(self.plan),
            "actions": copy.deepcopy(self.actions),
            "notes": copy.deepcopy(self.notes),
        }

    def _restore_snapshot(self, snap: dict):
        """Replace live state from a branch snapshot."""
        self.stm.messages = copy.deepcopy(snap.get("history", []))
        self.stm.summary = snap.get("summary", "")
        self.facts = copy.deepcopy(snap.get("facts", {}))
        self.stage = snap.get("stage", "IDLE")
        self.goal = snap.get("goal", "")
        self.plan = copy.deepcopy(snap.get("plan", []))
        self.actions = copy.deepcopy(snap.get("actions", []))
        self.notes = copy.deepcopy(snap.get("notes", []))

    def checkpoint(self):
        """Save current live state as a snapshot of the current branch."""
        self.branches[self.current_branch] = self._snapshot()

    def branch_create(self, name: str):
        """Create a new branch from current state. Raises ValueError if name exists."""
        if name in self.branches:
            raise ValueError(f"Branch '{name}' already exists")
        self.checkpoint()
        self.branches[name] = self._snapshot()

    def branch_switch(self, name: str):
        """Switch to an existing branch, saving current state first."""
        if name not in self.branches:
            raise ValueError(f"Branch '{name}' does not exist")
        self.checkpoint()
        self.current_branch = name
        self._restore_snapshot(self.branches[name])

    def reset(self):
        """Reset all state."""
        self.stage = "IDLE"
        self.goal = ""
        self.plan = []
        self.actions = []
        self.notes = []
        self.stm.messages = []
        self.stm.summary = ""
        self.facts = {}
        self.current_branch = "main"
        self.branches = {"main": self._snapshot()}

    def set_goal(self, goal: str):
        """Set global goal."""
        self.goal = goal

    def set_stage(self, stage: str):
        """Set current stage."""
        if stage not in ("IDLE", "PLAN", "EXECUTE", "REVIEW"):
            raise ValueError("Invalid stage")
        self.stage = stage

    def set_system_prompt(self, text: str):
        """Override system prompt."""
        self.system_prompt = text

    # ---------------- Persistence ----------------

    def _state_dict(self) -> dict:
        """Return working state as dict (no dialogue history or summary)."""
        return {
            "stage": self.stage,
            "goal": self.goal,
            "plan": self.plan,
            "actions": self.actions,
            "notes": self.notes,
            "facts": self.facts,
            "current_branch": self.current_branch,
            "branches": self.branches,
        }

    def save_state(self, path: str):
        """Save working state to TOON file."""
        with open(path, "w", encoding="utf-8") as f:
            toons.dump(self._state_dict(), f)

    def save_short_term(self, path: str):
        """Save short-term memory to a separate TOON file."""
        with open(path, "w", encoding="utf-8") as f:
            toons.dump(self.stm.to_dict(), f)

    def load_short_term(self, path: str):
        """Load short-term memory from a separate TOON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = toons.load(f)
        self.stm.from_dict(data)

    def load_state(self, path: str):
        """Load working state from TOON file."""
        with open(path, "r", encoding="utf-8") as f:
            st = toons.load(f)
        self.stage = st.get("stage", "IDLE")
        self.goal = st.get("goal", "")
        self.plan = st.get("plan", []) or []
        self.actions = st.get("actions", []) or []
        self.notes = st.get("notes", []) or []
        self.facts = st.get("facts", {}) or {}
        self.current_branch = st.get("current_branch", "main") or "main"
        self.branches = st.get("branches", {}) or {}
        # Migrate old state files that have no branches
        if not self.branches:
            self.branches = {"main": self._snapshot()}
            self.current_branch = "main"

    # ---------------- Prompt building ----------------
    def _select_context_messages(self) -> list[dict]:
        """
        Select which messages to include in the prompt depending on context strategy.
        Currently implemented:
          - window: Sliding Window (last N messages)
          - facts: Sliding Window + FACTS block (injected in _build_prompt)
          - branch: Sliding Window within the active branch (branches managed via CLI)
        """
        if not self.history_limit:
            return []
        # All strategies use a sliding window over the active short-term messages
        return self.stm.messages[-self.history_limit:]


    def _update_facts(self, user_text: str):
        """Parse 'Key: Value' lines from user message and update facts store."""
        for line in user_text.splitlines():
            if ": " in line:
                key, _, value = line.partition(": ")
                key = key.strip()
                value = value.strip()
                if key:
                    self.facts[key] = value

    def _build_prompt(self, user_text: str) -> str:
        """Build full prompt with state and history."""
        parts = []
        parts.append("SYSTEM:\n" + self.system_prompt.strip())
        state_toon = toons.dumps(
            {
                "stage": self.stage,
                "goal": self.goal,
                "plan": self.plan,
                "actions": self.actions,
                "notes": self.notes,
            }
        )
        parts.append("\nSTATE (TOON v3.0):\n" + state_toon.strip())

        if self.context_summary and self.stm.summary:
            parts.append("\nSUMMARY:\n" + self.stm.summary.strip())

        if self.context_strategy == "facts" and self.facts:
            facts_lines = "\n".join(f"{k}={v}" for k, v in self.facts.items())
            parts.append("\nFACTS:\n" + facts_lines)

        parts.append("\nDIALOG:")
        recent = self._select_context_messages()
        for m in recent:
            role = m["role"]
            text = m["text"]
            if role == "user":
                parts.append(f"User: {text}")
            else:
                parts.append(f"Assistant: {text}")
        parts.append("Assistant:")
        return "\n".join(parts)

    # ---------------- Summary compression ----------------

    def _summarize_messages(self, messages: list[dict]) -> str:
        """Summarize a chunk of dialog messages into short Russian notes."""
        chunk = "\n".join(f"{m['role']}: {m['text']}" for m in messages)
        prompt = (
            "Ты — модуль сжатия истории диалога.\n"
            "Обнови summary на русском, добавив важное из нового фрагмента.\n"
            "Требования:\n"
            "- 5–12 коротких пунктов\n"
            "- сохранить факты, цели, решения, договорённости, требования\n"
            "- без воды, без выдумок\n\n"
            "Текущее summary (может быть пустым):\n"
            f"{self.stm.summary}\n\n"
            "Новый фрагмент диалога:\n"
            f"{chunk}\n\n"
            "Верни ТОЛЬКО обновлённое summary:"
        )

        payload = {
            "model": self.model,
            "input": prompt,
            "temperature": 0,
            "max_output_tokens": 250,
        }
        data, _elapsed = self._post(payload)

        if isinstance(data, dict) and data.get("error") is not None:
            err = data.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(msg or "API error (summary)")

        try:
            return data["output"][0]["content"][0]["text"].strip()
        except Exception:
            return (self.stm.summary or "").strip()

    def _compress_history_if_needed(self):
        if not self.context_summary:
            return
        if not self.history_limit or self.history_limit <= 0:
            return

        overflow = len(self.stm.messages) - self.history_limit
        if overflow >= self.stm.summary_chunk_size:
            take = self.stm.summary_chunk_size
            chunk = self.stm.messages[:take]

            print("Сжатие истории...")
            self.stm.summary = self._summarize_messages(chunk)
            del self.stm.messages[:take]

    # ---------------- API call ----------------

    def _post(self, payload: dict):
        """Send request with retry."""
        retries = 3
        delay = 2
        start = time.monotonic()
        for i in range(retries):
            try:
                r = requests.post(
                    URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                elapsed = time.monotonic() - start
                return r.json(), elapsed
            except requests.exceptions.ReadTimeout:
                if i == retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        return {}, 0.0

    # ---------------- Main interaction ----------------

    def reply(self, user_text: str):
        """Send user message to LLM and return reply + metrics."""
        self.stm.messages.append({"role": "user", "text": user_text})

        if self.context_strategy == "facts":
            self._update_facts(user_text)

        # Day 9: compress history before building prompt
        self._compress_history_if_needed()

        prompt = self._build_prompt(user_text)

        payload = {
            "model": self.model,
            "input": prompt,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            payload["max_output_tokens"] = self.max_output_tokens
        if self.stop:
            payload["stop"] = self.stop

        data, elapsed = self._post(payload)

        if isinstance(data, dict) and data.get("error") is not None:
            err = data.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(msg or "API error")

        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected API response type: {type(data)}")

        if self.print_json:
            text = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            try:
                text = data["output"][0]["content"][0]["text"]
            except Exception:
                text = json.dumps(data, ensure_ascii=False, indent=2)

        self.stm.messages.append({"role": "assistant", "text": text})

        usage = data.get("usage", {})

        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")

        in_cost, out_cost, total_cost = compute_cost(
            self.pricing, self.model, in_tok, out_tok
        )

        metrics = {
            "model": self.model,
            "time": elapsed,
            "in": in_tok,
            "out": out_tok,
            "cost": total_cost,
        }

        return text, metrics
