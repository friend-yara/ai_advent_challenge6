#!/usr/bin/env python3
import copy
import json
import sys
import time
from pathlib import Path
from typing import Optional

import requests

try:
    import yaml
except ImportError as e:
    raise SystemExit(
        "ERROR: Install YAML support:\n"
        "  pip install pyyaml"
    ) from e

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


# Allowed TaskContext state transitions
TASK_TRANSITIONS: dict[str, list[str]] = {
    "PLANNING":   ["EXECUTION", "DONE"],
    "EXECUTION":  ["VALIDATION", "PLANNING", "DONE"],
    "VALIDATION": ["PLANNING", "EXECUTION", "DONE"],
    "DONE":       ["PLANNING"],
}

# Mapping from legacy stage names to TaskContext states
_STAGE_ALIAS: dict[str, str] = {
    "IDLE":    "PLANNING",
    "PLAN":    "PLANNING",
    "EXECUTE": "EXECUTION",
    "REVIEW":  "VALIDATION",
}


class Profile:
    """
    User profile loaded from a JSON file.
    Sections style/constraints/context are rendered as YAML for prompt injection.
    Injection is controlled by the prompt_injection section in the JSON.
    """

    # Sections excluded from YAML rendering (internal/meta)
    _EXCLUDED_SECTIONS = ("meta", "prompt_injection")

    def __init__(self, path: str):
        """Initialize profile with file path. Empty path disables the profile."""
        self.path = path
        self.data: dict = {}
        self._loaded: bool = False

    def load(self) -> str | None:
        """Read and parse the JSON file. Returns error string or None on success."""
        if not self.path:
            return None
        p = Path(self.path)
        if not p.exists():
            return f"Profile file not found: {self.path}"
        try:
            self.data = json.loads(p.read_text(encoding="utf-8"))
            self._loaded = True
            return None
        except Exception as e:
            return f"Could not load profile {self.path}: {e}"

    def reload(self) -> str | None:
        """Re-read the file from disk."""
        self.data = {}
        self._loaded = False
        return self.load()

    @property
    def enabled(self) -> bool:
        """True if prompt_injection.enabled is set (default True if section missing)."""
        pi = self.data.get("prompt_injection", {})
        return bool(self._loaded and pi.get("enabled", True))

    @property
    def inject_as(self) -> str:
        """Block label for prompt injection (from prompt_injection.inject_as)."""
        pi = self.data.get("prompt_injection", {})
        return pi.get("inject_as", "PROFILE")

    def to_yaml(self) -> str:
        """Render style/constraints/context sections as YAML string."""
        renderable = {
            k: v for k, v in self.data.items()
            if k not in self._EXCLUDED_SECTIONS
        }
        return yaml.dump(
            renderable,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ).strip()

    def summary_line(self) -> str:
        """One-line status for /show."""
        if not self._loaded:
            return "not loaded"
        meta = self.data.get("meta", {})
        profile_id = meta.get("id", "?")
        status = "enabled" if self.enabled else "disabled"
        return f"id={profile_id}, {status}"


class LongTermMemory:
    """
    Long-term memory: profile, invariants, project context.
    Loaded from md/txt files; never saved to state.toon or short_term.toon.
    Cached in process memory; reloaded on demand via reload().
    """

    def __init__(
        self,
        project_memory_file: str = "PROJECT_MEMORY.md",
        profile_file: str = "PROFILE.md",
        invariants_file: str = "INVARIANTS.md",
        use_project_memory: bool = False,
        use_profile: bool = False,
        use_invariants: bool = False,
    ):
        """Initialize LTM with file paths and injection flags."""
        self.project_memory_file = project_memory_file
        self.profile_file = profile_file
        self.invariants_file = invariants_file
        self.use_project_memory = use_project_memory
        self.use_profile = use_profile
        self.use_invariants = use_invariants

        # Cached content
        self.project_memory: str | None = None
        self.profile_obj: Profile | None = None
        self.invariants: str | None = None

    def _read_file(self, path: str) -> str | None:
        """Read a text file, returning None with a warning if missing."""
        p = Path(path)
        if not p.exists():
            print(f"[WARN] LTM file not found: {path}", file=sys.stderr)
            return None
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"[WARN] Could not read LTM file {path}: {e}", file=sys.stderr)
            return None

    def load(self):
        """Load all enabled LTM files into cache."""
        if self.use_project_memory:
            self.project_memory = self._read_file(self.project_memory_file)
        if self.use_profile:
            self.profile_obj = Profile(self.profile_file)
            err = self.profile_obj.load()
            if err:
                print(f"[WARN] {err}", file=sys.stderr)
                self.profile_obj = None
        if self.use_invariants:
            self.invariants = self._read_file(self.invariants_file)

    def reload(self):
        """Re-read LTM files from disk (clears cache first)."""
        self.project_memory = None
        self.profile_obj = None
        self.invariants = None
        self.load()

    def blocks(self, task_state: str) -> list[tuple[str, str]]:
        """
        Return list of (block_name, content) to inject into the prompt.
        Injection rules:
          - PROFILE: always, if loaded and enabled (rendered as YAML)
          - PROJECT_MEMORY: always, if loaded
          - INVARIANTS: always if loaded; mandatory when task_state == EXECUTION
        Returns only non-empty blocks.
        """
        result = []
        if self.use_profile and self.profile_obj and self.profile_obj.enabled:
            result.append((self.profile_obj.inject_as, self.profile_obj.to_yaml()))
        if self.use_project_memory and self.project_memory:
            result.append(("PROJECT_MEMORY", self.project_memory))
        if self.use_invariants and self.invariants:
            result.append(("INVARIANTS", self.invariants))
        return result

    def summary_line(self) -> str:
        """One-line status for /show."""
        parts = []
        if self.use_project_memory:
            parts.append(f"project={'yes' if self.project_memory else 'not loaded'}")
        if self.use_profile:
            p_status = self.profile_obj.summary_line() if self.profile_obj else "not loaded"
            parts.append(f"profile={p_status}")
        if self.use_invariants:
            parts.append(f"invariants={'yes' if self.invariants else 'not loaded'}")
        return ", ".join(parts) if parts else "disabled"


class TaskContext:
    """Working memory: current task data (state machine, plan, progress)."""

    VALID_STATES = tuple(TASK_TRANSITIONS.keys())

    def __init__(self):
        """Initialize with blank task in PLANNING state."""
        self.task: str = ""
        self.state: str = "PLANNING"
        self.step: int = 0
        self.total: int = 0
        self.plan: list[str] = []
        self.done: list[str] = []
        self.current: str = ""
        # Backward-compat fields
        self.actions: list[str] = []
        self.notes: list[str] = []

    def set_state(self, new_state: str) -> str | None:
        """Transition to new_state. Returns None on success, error string on failure."""
        new_state = new_state.upper()
        if new_state not in TASK_TRANSITIONS:
            return f"Invalid state '{new_state}'. Valid: {', '.join(TASK_TRANSITIONS)}"
        allowed = TASK_TRANSITIONS.get(self.state, [])
        if new_state not in allowed:
            return (
                f"Transition {self.state} → {new_state} not allowed. "
                f"Allowed from {self.state}: {', '.join(allowed) or 'none'}"
            )
        self.state = new_state
        return None

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "task": self.task,
            "state": self.state,
            "step": self.step,
            "total": self.total,
            "plan": self.plan,
            "done": self.done,
            "current": self.current,
            "actions": self.actions,
            "notes": self.notes,
        }

    def from_dict(self, data: dict):
        """Restore from plain dict."""
        self.task = data.get("task", "") or ""
        self.state = data.get("state", "PLANNING") or "PLANNING"
        self.step = data.get("step", 0) or 0
        self.total = data.get("total", 0) or 0
        self.plan = data.get("plan", []) or []
        self.done = data.get("done", []) or []
        self.current = data.get("current", "") or ""
        self.actions = data.get("actions", []) or []
        self.notes = data.get("notes", []) or []


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
        ltm: "LongTermMemory | None" = None,
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

        # Day 11: long-term memory layer (loaded externally, injected on demand)
        self.ltm: LongTermMemory = ltm or LongTermMemory()

        # Day 11: short-term memory layer
        self.stm = ShortTermMemory(
            history_limit=self.history_limit,
            summary_chunk_size=10,
        )

        # Day 11: working memory layer
        self.tc = TaskContext()

        # Persistent state (working memory — remaining fields)
        self.facts: dict[str, str] = {}
        self.current_branch: str = "main"
        self.branches: dict[str, dict] = {"main": self._snapshot()}

    # ---------------- State management ----------------

    # Backward-compat properties delegating to TaskContext
    @property
    def stage(self) -> str:
        """Current task state (delegates to tc.state)."""
        return self.tc.state

    @stage.setter
    def stage(self, value: str):
        self.tc.state = value

    @property
    def goal(self) -> str:
        """Current task description (delegates to tc.task)."""
        return self.tc.task

    @goal.setter
    def goal(self, value: str):
        self.tc.task = value

    @property
    def plan(self) -> list:
        """Current plan steps (delegates to tc.plan)."""
        return self.tc.plan

    @plan.setter
    def plan(self, value: list):
        self.tc.plan = value

    @property
    def actions(self) -> list:
        """Completed actions (delegates to tc.actions)."""
        return self.tc.actions

    @actions.setter
    def actions(self, value: list):
        self.tc.actions = value

    @property
    def notes(self) -> list:
        """Notes (delegates to tc.notes)."""
        return self.tc.notes

    @notes.setter
    def notes(self, value: list):
        self.tc.notes = value

    def _snapshot(self) -> dict:
        """Return a deep copy of current per-branch state (working + stm)."""
        return {
            "tc": copy.deepcopy(self.tc.to_dict()),
            "history": copy.deepcopy(self.stm.messages),
            "summary": self.stm.summary,
            "facts": copy.deepcopy(self.facts),
        }

    def _restore_snapshot(self, snap: dict):
        """Replace live state from a branch snapshot."""
        self.stm.messages = copy.deepcopy(snap.get("history", []))
        self.stm.summary = snap.get("summary", "")
        self.facts = copy.deepcopy(snap.get("facts", {}))
        if "tc" in snap:
            self.tc.from_dict(snap["tc"])
        else:
            # Migrate old snapshot format
            self.tc.task = snap.get("goal", "")
            old_stage = snap.get("stage", "PLANNING")
            self.tc.state = _STAGE_ALIAS.get(old_stage) or old_stage
            self.tc.plan = copy.deepcopy(snap.get("plan", []))
            self.tc.actions = copy.deepcopy(snap.get("actions", []))
            self.tc.notes = copy.deepcopy(snap.get("notes", []))

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
        self.tc = TaskContext()
        self.stm.messages = []
        self.stm.summary = ""
        self.facts = {}
        self.current_branch = "main"
        self.branches = {"main": self._snapshot()}

    def set_goal(self, goal: str):
        """Set task description (delegates to tc.task)."""
        self.tc.task = goal

    def set_task_state(self, state: str) -> str | None:
        """Set TaskContext state with transition validation. Returns error string or None."""
        return self.tc.set_state(state.upper())

    def set_stage(self, stage: str) -> str | None:
        """Set stage with legacy alias support and transition validation. Returns error string or None."""
        mapped = _STAGE_ALIAS.get(stage.upper(), stage.upper())
        return self.tc.set_state(mapped)

    def set_system_prompt(self, text: str):
        """Override system prompt."""
        self.system_prompt = text

    # ---------------- Persistence ----------------

    def _state_dict(self) -> dict:
        """Return working state as dict (TaskContext + facts + branches, no dialogue)."""
        return {
            "tc": self.tc.to_dict(),
            "facts": self.facts,
            "current_branch": self.current_branch,
            "branches": self.branches,
        }

    def save_state(self, path: str):
        """Save working state to TOON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            toons.dump(self._state_dict(), f)

    def save_short_term(self, path: str):
        """Save short-term memory to a separate TOON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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
        if "tc" in st:
            self.tc.from_dict(st["tc"])
        else:
            # Migrate old flat format (stage/goal/plan/actions/notes)
            self.tc.task = st.get("goal", "")
            old_stage = st.get("stage", "PLANNING")
            self.tc.state = _STAGE_ALIAS.get(old_stage) or old_stage
            self.tc.plan = st.get("plan", []) or []
            self.tc.actions = st.get("actions", []) or []
            self.tc.notes = st.get("notes", []) or []
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

        # Long-term memory blocks (injected before STATE, after SYSTEM)
        for block_name, content in self.ltm.blocks(self.tc.state):
            parts.append(f"\n{block_name}:\n{content}")

        parts.append("\nSTATE (TOON v3.0):\n" + toons.dumps(self.tc.to_dict()).strip())

        if self.tc.current:
            parts.append(
                "\nRULES:\n"
                f"- Work only within the current step: {self.tc.current}\n"
                "- Do not skip steps\n"
                "- When step is complete, signal next_step"
            )

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

    # ---------------- Profile summary ----------------

    def whoami(self, profile_name: str) -> str:
        """Return a short LLM-generated summary of the current user profile."""
        sections = []
        if self.ltm.profile_obj and self.ltm.profile_obj.enabled:
            sections.append("PROFILE:\n" + self.ltm.profile_obj.to_yaml())
        if self.ltm.project_memory:
            sections.append("PROJECT_MEMORY:\n" + self.ltm.project_memory)
        if self.ltm.invariants:
            sections.append("INVARIANTS:\n" + self.ltm.invariants)

        if not sections:
            return f"Профиль «{profile_name}»: данные не загружены."

        context = "\n\n".join(sections)
        prompt = (
            f"Ты — ассистент. На основе данных профиля пользователя "
            f"«{profile_name}» напиши краткое описание на русском языке. "
            f"Максимум 80 слов. Только факты из данных, без выдумок.\n\n"
            f"{context}"
        )
        payload = {
            "model": self.model,
            "input": prompt,
            "temperature": 0,
            "max_output_tokens": 150,
        }
        data, _ = self._post(payload)
        if isinstance(data, dict) and data.get("error"):
            err = data.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(msg or "API error (whoami)")
        try:
            return data["output"][0]["content"][0]["text"].strip()
        except Exception:
            return f"Профиль «{profile_name}»: не удалось получить ответ."

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
