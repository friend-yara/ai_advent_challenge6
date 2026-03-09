#!/usr/bin/env python3
"""
agents.py — AgentSpec dataclass and AgentRegistry.

Loads agent definitions from Markdown files with YAML front-matter.
Each file in agents/ defines one agent: model, context policy, prompt, etc.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError as e:
    raise SystemExit("ERROR: Install YAML support: pip install pyyaml") from e


@dataclass
class ContextPolicy:
    """Declares which context layers an agent receives."""

    include_profile: bool = False
    include_invariants: bool = False
    include_project_memory: bool = False
    include_state: bool = True
    include_history: bool = False
    history_limit: int = 4
    include_rules_block: bool = False
    include_validation_block: bool = False
    include_summary: bool = False
    include_facts: bool = False
    include_task: bool = False       # inject only tc.task (no full state)
    include_plan_summary: bool = False  # inject plan+done as a block

    @classmethod
    def from_dict(cls, data: dict) -> "ContextPolicy":
        """Build from a plain dict (YAML front-matter context_policy section)."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class AgentSpec:
    """
    Specification for a single agent loaded from an agents/*.md file.
    Fields match the YAML front-matter keys.
    """

    name: str
    mode: str                        # "primary" | "subagent"
    description: str
    model: str
    temperature: float | None
    when_to_use: str
    allowed_states: list[str]
    context_policy: ContextPolicy
    prompt: str                      # resolved system prompt text
    source_file: str = ""            # path to the .md file (informational)

    def allows_state(self, state: str) -> bool:
        """Return True if this agent is allowed to run in the given state."""
        return state.upper() in [s.upper() for s in self.allowed_states]


class AgentRegistry:
    """
    Loads and indexes AgentSpec objects from a directory of *.md files.

    Each file must start with a YAML front-matter block delimited by ---.
    The text after the closing --- is the agent's system prompt.
    """

    def __init__(self):
        """Initialize with empty registry."""
        self._specs: dict[str, AgentSpec] = {}

    def load(self, agents_dir: str | Path):
        """
        Scan agents_dir for *.md files and load each as an AgentSpec.
        Skips files with parse errors (prints WARN to stderr).
        """
        agents_dir = Path(agents_dir)
        if not agents_dir.is_dir():
            print(f"[WARN] agents dir not found: {agents_dir}", file=sys.stderr)
            return
        for path in sorted(agents_dir.glob("*.md")):
            spec = self._load_file(path)
            if spec is not None:
                self._specs[spec.name] = spec

    def _load_file(self, path: Path) -> "AgentSpec | None":
        """Parse a single agent spec file. Returns None on error."""
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Cannot read agent spec {path}: {e}", file=sys.stderr)
            return None

        # Split YAML front-matter from prompt body
        front_matter, prompt_text = _split_front_matter(text)
        if front_matter is None:
            print(f"[WARN] No YAML front-matter in {path}", file=sys.stderr)
            return None

        try:
            meta = yaml.safe_load(front_matter) or {}
        except yaml.YAMLError as e:
            print(f"[WARN] YAML parse error in {path}: {e}", file=sys.stderr)
            return None

        required = ("name", "mode", "description", "model", "allowed_states")
        missing = [k for k in required if k not in meta]
        if missing:
            print(
                f"[WARN] Agent spec {path} missing required keys: {missing}",
                file=sys.stderr,
            )
            return None

        policy_data = meta.get("context_policy", {}) or {}
        policy = ContextPolicy.from_dict(policy_data)

        return AgentSpec(
            name=meta["name"],
            mode=meta.get("mode", "primary"),
            description=meta.get("description", ""),
            model=meta["model"],
            temperature=meta.get("temperature"),
            when_to_use=meta.get("when_to_use", ""),
            allowed_states=meta.get("allowed_states", []),
            context_policy=policy,
            prompt=prompt_text.strip(),
            source_file=str(path),
        )

    def get(self, name: str) -> "AgentSpec | None":
        """Return spec by name, or None if not found."""
        return self._specs.get(name)

    def for_state(self, state: str) -> "AgentSpec | None":
        """
        Return the first primary agent whose allowed_states includes state.
        Returns None if no match found.
        """
        for spec in self._specs.values():
            if spec.mode == "primary" and spec.allows_state(state):
                return spec
        return None

    def list_all(self) -> list[AgentSpec]:
        """Return all loaded specs sorted by name."""
        return sorted(self._specs.values(), key=lambda s: s.name)

    def list_primaries(self) -> list[AgentSpec]:
        """Return only primary agents sorted by name."""
        return [s for s in self.list_all() if s.mode == "primary"]

    def summary(self) -> str:
        """One-line summary: count of loaded agents."""
        n = len(self._specs)
        names = ", ".join(self._specs.keys())
        return f"{n} agent(s): {names}"


# ---------------- Helpers ----------------

def _split_front_matter(text: str) -> tuple[str | None, str]:
    """
    Split a Markdown file into (yaml_front_matter, body).
    Front-matter is the content between the first and second '---' lines.
    Returns (None, full_text) if no valid front-matter found.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text

    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = i
            break

    if end is None:
        return None, text

    front = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1:])
    return front, body
