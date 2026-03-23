#!/usr/bin/env python3
import copy
import json
import re
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

from providers import OpenAIProvider


def load_pricing_models(script_dir: Optional[Path] = None) -> dict:
    """Load pricing models from pricing.json."""
    base = script_dir or Path(__file__).resolve().parent
    path = base / "pricing.json"

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("models", {}) or {}
    except Exception:
        print("[WARN] pricing.json not found, cost calculation disabled")
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


# Allowed TaskContext state transitions (strict graph)
TASK_TRANSITIONS: dict[str, list[str]] = {
    "CHAT":       ["PLANNING"],
    "PLANNING":   ["EXECUTION", "CHAT"],
    "EXECUTION":  ["VALIDATION", "PLANNING"],
    "VALIDATION": ["DONE", "EXECUTION"],
    "DONE":       ["PLANNING", "CHAT"],
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


class Invariants:
    """
    Project invariants loaded from a YAML file.
    Rendered as structured text for prompt injection.
    Banned list is passed to InvariantChecker for rule enforcement.
    """

    _EXCLUDED_SECTIONS = ("meta",)

    def __init__(self, path: str):
        """Initialize with file path."""
        self.path = path
        self.data: dict = {}
        self._loaded: bool = False

    def load(self) -> str | None:
        """Read and parse YAML file. Returns error string or None on success."""
        if not self.path:
            return None
        p = Path(self.path)
        if not p.exists():
            return f"Invariants file not found: {self.path}"
        try:
            self.data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            self._loaded = True
            return None
        except Exception as e:
            return f"Could not load invariants {self.path}: {e}"

    def reload(self) -> str | None:
        """Re-read the file from disk."""
        self.data = {}
        self._loaded = False
        return self.load()

    def to_text(self) -> str:
        """
        Render invariants as human-readable text for prompt injection.
        Excludes the meta section. Each section becomes a header + items.
        """
        lines = []
        for section, value in self.data.items():
            if section in self._EXCLUDED_SECTIONS:
                continue
            header = section.replace("_", " ").title()
            lines.append(f"{header}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, list):
                        lines.append(f"  {k}:")
                        for item in v:
                            lines.append(f"    - {item}")
                    else:
                        lines.append(f"  {k}: {v}")
            elif isinstance(value, list):
                for item in value:
                    label = item.get("rule", item) if isinstance(item, dict) else item
                    lines.append(f"  - {label}")
            else:
                lines.append(f"  {value}")
            lines.append("")
        return "\n".join(lines).strip()

    def banned_items(self) -> list[dict]:
        """Return banned rules as list of {rule, patterns} dicts."""
        return self.data.get("banned", [])


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
        self.invariants_obj: Invariants | None = None
        self.invariants: str | None = None   # rendered text cache
        self.checker: InvariantChecker = InvariantChecker()

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
            self.invariants_obj = Invariants(self.invariants_file)
            err = self.invariants_obj.load()
            if err:
                print(f"[WARN] {err}", file=sys.stderr)
                self.invariants_obj = None
            else:
                self.invariants = self.invariants_obj.to_text()
                self.checker.load_from_items(self.invariants_obj.banned_items())

    def reload(self):
        """Re-read LTM files from disk (clears cache first)."""
        self.project_memory = None
        self.profile_obj = None
        self.invariants_obj = None
        self.invariants = None
        self.checker = InvariantChecker()
        self.load()

    def summary_line(self) -> str:
        """One-line status for /show."""
        parts = []
        if self.use_project_memory:
            parts.append(f"project={'yes' if self.project_memory else 'not loaded'}")
        if self.use_profile:
            p_status = self.profile_obj.summary_line() if self.profile_obj else "not loaded"
            parts.append(f"profile={p_status}")
        if self.use_invariants:
            if self.invariants_obj and self.invariants:
                inv_status = self.checker.summary_line()
            else:
                inv_status = "not loaded"
            parts.append(f"invariants={inv_status}")
        return ", ".join(parts) if parts else "disabled"


class InvariantRule:
    """A single invariant rule with human-readable description and regex patterns."""

    def __init__(self, description: str, patterns: list[str]):
        """Compile regex patterns at init time."""
        self.description = description
        self.compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    def matches(self, text: str) -> bool:
        """Return True if any pattern matches the text."""
        return any(p.search(text) for p in self.compiled)


class InvariantChecker:
    """
    Checks text (user queries, LLM answers) against loaded invariant rules.
    Rules are parsed from INVARIANTS.yaml at load time. No extra API calls.
    """

    def __init__(self):
        """Initialize with empty rule set."""
        self.rules: list[InvariantRule] = []
        self.last_result: tuple[bool, list[str]] = (True, [])

    def load_from_items(self, items: list[dict]):
        """Load rules from list of {rule, patterns} dicts (from INVARIANTS.yaml)."""
        self.rules = []
        for item in items:
            if not isinstance(item, dict):
                continue  # skip plain strings (old format)
            description = item.get("rule", "")
            patterns = item.get("patterns", [])
            if description and patterns:
                self.rules.append(InvariantRule(description, patterns))

    def check(self, text: str) -> tuple[bool, list[str]]:
        """
        Check text against all rules.
        Returns (passed, violations) where violations is list of descriptions.
        """
        violations = [r.description for r in self.rules if r.matches(text)]
        passed = not violations
        self.last_result = (passed, violations)
        return passed, violations

    def summary_line(self) -> str:
        """One-line status: rule count + last check result."""
        passed, violations = self.last_result
        status = "PASS" if passed else f"FAIL({len(violations)})"
        return f"rules={len(self.rules)}, last={status}"


class TaskContext:
    """Working memory: current task data (state machine, plan, progress)."""

    VALID_STATES = tuple(TASK_TRANSITIONS.keys())

    def __init__(self):
        """Initialize with blank task in CHAT state."""
        self.task: str = ""
        self.state: str = "CHAT"
        self.step: int = 0
        self.total: int = 0
        self.plan: list[str] = []
        self.done: list[str] = []
        self.current: str = ""
        # Backward-compat fields
        self.actions: list[str] = []
        self.notes: list[str] = []

    def can_transition_to_execution(self) -> bool:
        """Return True only when plan is non-empty."""
        return bool(self.plan)

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
        self.state = data.get("state", "CHAT") or "CHAT"
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
    Thin LLM caller with memory layers and TOON persistence.

    Responsibilities:
    - HTTP calls to OpenAI Responses API (_post)
    - State machine and persistence (TaskContext, TOON files)
    - Short-term and long-term memory storage
    - History compression (_compress_history_if_needed)
    - Utility methods: welcome_back, whoami, format_todo, plan_from_reply

    Prompt building and agent routing are handled by ContextBuilder +
    Orchestrator — not by this class.
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
        provider=None,
    ):
        """Initialize agent."""
        self.api_key = api_key
        self.provider = provider or OpenAIProvider(api_key)
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

        # Summary compression flag (config, not state)
        self.context_summary = context_summary

        # Long-term memory layer (loaded externally, injected on demand)
        self.ltm: LongTermMemory = ltm or LongTermMemory()

        # Short-term memory layer
        self.stm = ShortTermMemory(
            history_limit=self.history_limit,
            summary_chunk_size=10,
        )

        # Working memory layer
        self.tc = TaskContext()

        # Persistent state (working memory — remaining fields)
        self.facts: dict[str, str] = {}
        self.current_branch: str = "main"
        self.branches: dict[str, dict] = {"main": self._snapshot()}

    # ---------------- State management ----------------

    # Backward-compat properties delegating to TaskContext
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
        self.facts = st.get("facts", {}) or {}
        self.current_branch = st.get("current_branch", "main") or "main"
        self.branches = st.get("branches", {}) or {}
        # Migrate old state files that have no branches
        if not self.branches:
            self.branches = {"main": self._snapshot()}
            self.current_branch = "main"

    # ---------------- Task state machine helpers ----------------

    def reset_working(self):
        """Reset working memory (TaskContext) only. Does not touch STM."""
        self.tc = TaskContext()
        self.facts = {}
        self.current_branch = "main"
        self.branches = {"main": self._snapshot()}

    def reset_short_term(self):
        """Reset short-term memory only. Does not touch working memory."""
        self.stm.messages = []
        self.stm.summary = ""

    def format_todo(self) -> str:
        """Return current plan as a [ ]/[x] checklist string."""
        if not self.tc.plan:
            return "(план пуст)"
        lines = []
        for i, step in enumerate(self.tc.plan):
            mark = "x" if i < self.tc.step else " "
            lines.append(f"[{mark}] {step}")
        return "\n".join(lines)

    def plan_from_reply(self, text: str) -> list[str] | None:
        """
        Parse a TODO checklist from LLM reply.
        Recognises lines matching '[ ] ...', '- [ ] ...', or '1. ...' patterns.
        Returns list of step strings or None if no list detected.
        """
        steps = []
        for line in text.splitlines():
            line = line.strip()
            # [ ] step or - [ ] step
            m = re.match(r"[-*]?\s*\[\s*\]\s+(.+)", line)
            if m:
                steps.append(m.group(1).strip())
                continue
            # 1. step or 1) step
            m = re.match(r"\d+[.)]\s+(.+)", line)
            if m:
                steps.append(m.group(1).strip())
        return steps if len(steps) >= 2 else None

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

    def welcome_back(self) -> str:
        """
        Generate a short 'welcome back' message using last STM messages + tc state.
        Falls back to plain text if LLM call fails.
        """
        last_msgs = self.stm.messages[-2:] if self.stm.messages else []
        context_lines = "\n".join(
            f"{m['role']}: {m['text']}" for m in last_msgs
        )
        step_info = ""
        if self.tc.state == "EXECUTION" and self.tc.plan:
            step_info = (
                f"Шаг {self.tc.step}/{self.tc.total}: "
                f"{self.tc.current or '—'}"
            )
        elif self.tc.task:
            step_info = f"Задача: {self.tc.task}"

        prompt = (
            "Пользователь вернулся в агент после перерыва. "
            "Напиши одно-два предложения приветствия на русском языке, "
            "начиная с 'С возвращением!', и кратко напомни, "
            "на чём остановились.\n\n"
            f"Состояние: {self.tc.state}\n"
            f"{step_info}\n"
            f"Последний диалог:\n{context_lines or '(нет)'}"
        )
        try:
            payload = {
                "model": self.model,
                "input": prompt,
                "temperature": 0,
                "max_output_tokens": 100,
            }
            data, _ = self._post(payload)
            if isinstance(data, dict) and not data.get("error"):
                return data["output"][0]["content"][0]["text"].strip()
        except Exception:
            pass
        # Fallback to plain text
        return (
            f"С возвращением! Вы остановились на этапе {self.tc.state}"
            + (f": {step_info}" if step_info else ".")
        )

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
        """Compress STM history into summary when overflow threshold is reached."""
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
        """Send request to the configured LLM provider."""
        return self.provider.post(payload, self.timeout)
