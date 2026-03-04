# AGENTS.md — AI Advent Challenge

Guidance for agentic coding agents operating in this repository.

---

## Project Overview

An exploratory AI/LLM learning project (AI Advent Challenge). Pure Python CLI
application implementing an agent loop against the OpenAI Responses API — no
SDK, no web framework.

- **Language:** Python 3.12
- **Platform:** Ubuntu 24.04, SSH-based VDS
- **Entry point:** `llm_agent_cli.py` (REPL) / `agent.py` (core logic)

---

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

Dependencies are fully pinned in `requirements.txt`. Do not install packages
system-wide (PEP 668 — venv only). After adding a dependency, pin its exact
version in `requirements.txt`.

---

## Running the Project

| Command | Description |
|---|---|
| `python llm_agent_cli.py` | Start agent REPL (default profile) |
| `python llm_agent_cli.py --profile alice` | Start with a named profile |
| `python llm_agent_cli.py -m gpt-4.1-mini` | Use a specific model |
| `python llm_agent_cli.py -t 0.7` | Set temperature |
| `python llm_agent_cli.py --history-limit 10` | Limit history window |
| `python llm_agent_cli.py --context-summary` | Enable context compression |
| `python llm_agent_cli.py --use-invariants` | Inject INVARIANTS into prompt |
| `python llm_agent_cli.py --use-project-memory` | Inject PROJECT_MEMORY |
| `llm_agent` | Same as above via symlink in ~/bin |

---

## Build / Lint / Test Commands

**There is no build step, no configured linter, and no test suite.**

- No `Makefile`, `pyproject.toml`, `tox.ini`, or `pytest.ini` exist.
- No test files exist. If you add tests, use `pytest` and place them in a
  `tests/` directory.
- If you add a linter (recommended: `ruff`), run it with `ruff check .`.
- If you add a formatter (recommended: `ruff format`), run it with
  `ruff format .`.

**Running a single test (if tests are added):**
```bash
pytest tests/test_agent.py::test_function_name -v
```

---

## Architecture

```
agent.py                  Core Agent class — LLM calls, memory, persistence
llm_agent_cli.py          CLI REPL — argparse, REPL loop, /commands
system_prompt.txt         External system prompt (loaded at runtime)
pricing.json              Manual pricing table for token cost calculation
requirements.txt          Pinned Python dependencies
profiles/
  default/
    PROFILE.json          Default user profile (JSON, rendered as YAML)
    INVARIANTS.md         Stack/arch/budget/banned constraints (plain text)
    PROJECT_MEMORY.md     Project context document (plain text)
    state.toon            Working memory — gitignored
    short_term.toon       Dialogue history — gitignored
  <name>/                 Additional user profiles (same structure)
```

---

## Memory Architecture (Day 11+)

Three independent memory layers, each with its own storage:

| Layer | Class | File | Contents |
|---|---|---|---|
| Short-term | `ShortTermMemory` | `short_term.toon` | messages, summary |
| Working | `TaskContext` | `state.toon` | task, state, plan, step, done |
| Long-term | `LongTermMemory` + `Profile` | `.md` / `.json` files | profile, invariants, project memory |

**Rules:**
- `state.toon` must NEVER contain dialogue (`messages`/`summary`).
- `short_term.toon` must NEVER contain `TaskContext` fields.
- LTM files are read-only at runtime; never written by the agent.
- Both `.toon` files are gitignored — never commit them.

**Prompt block order:**
```
SYSTEM → PROFILE → PROJECT_MEMORY → INVARIANTS → STATE → RULES → SUMMARY → FACTS → DIALOG
```

---

## Key Classes in `agent.py`

### `Profile`
Loads `PROFILE.json`; renders `style`/`constraints`/`context` as YAML via
`yaml.dump()` for prompt injection. `meta` and `prompt_injection` sections are
excluded from rendering. `prompt_injection.enabled` controls whether the block
is injected.

### `LongTermMemory`
Loads `Profile`, `PROJECT_MEMORY.md`, `INVARIANTS.md` into process cache.
Never persisted to disk. `blocks(task_state)` returns `(name, content)` pairs
to inject. `reload()` re-reads all files without restarting.

### `TaskContext`
Working memory state machine. States: `PLANNING → EXECUTION → VALIDATION →
DONE`. Transition validation is enforced in `set_state()` — invalid transitions
return an error string instead of raising. Legacy `IDLE/PLAN/EXECUTE/REVIEW`
names are mapped via `_STAGE_ALIAS`.

### `ShortTermMemory`
Current dialogue only (`messages` list + `summary` string). Serialised to
`short_term.toon` separately from working state.

### `Agent`
Composes all layers. `stage`/`goal`/`plan`/`actions`/`notes` are `@property`
shims delegating to `self.tc` for backward compatibility.

---

## API Usage

- **Endpoint:** `https://api.openai.com/v1/responses` (Responses API)
- **No OpenAI SDK** — all calls use `requests` with raw HTTP
- API key read from `OPENAI_API_KEY` environment variable
- Retry: 3 attempts, exponential backoff (2s → 4s → 8s) on `ReadTimeout`
- API errors detected via `data.get("error") is not None`, raised as
  `RuntimeError`

---

## Code Style

### Imports

Order: stdlib → third-party → local. One import per line. Use
`from pathlib import Path` for file operations. Wrap optional/required
third-party imports in `try/except ImportError` with `raise SystemExit`.

```python
import json
import sys
from pathlib import Path
from typing import Optional   # legacy; new code uses X | None

import requests

from agent import Agent, load_pricing_models
```

### Type Hints

- Use Python 3.10+ union syntax: `float | None`, `list[str] | None`
- Annotate all non-trivial function parameters and return types
- Use plain `dict` for API payloads — no `TypedDict`, `dataclasses`, `pydantic`

```python
def load_pricing_models(script_dir: Path | None = None) -> dict: ...
def set_state(self, new_state: str) -> str | None: ...
```

### Naming Conventions

| Entity | Convention | Example |
|---|---|---|
| Variables, functions, methods | `snake_case` | `history_limit` |
| Classes | `PascalCase` | `TaskContext` |
| Module-level constants | `SCREAMING_SNAKE_CASE` | `TASK_TRANSITIONS` |
| Private methods | `_underscore_prefix` | `_build_prompt` |
| CLI arguments | `--kebab-case` | `--history-limit` |

### Formatting

- f-strings for all string interpolation
- `json.dumps(..., ensure_ascii=False, indent=2)` for JSON output
- Section separators inside long files:
  ```python
  # ---------------- Persistence ----------------
  ```
- Aim for lines under 100 chars; no enforcer configured

### Docstrings

- One-line docstrings on all public methods and module-level functions
- Multi-line docstrings for classes (bullet list of capabilities)
- Private trivial helpers may omit docstrings

### Error Handling

- Catch specific exceptions: `requests.exceptions.ReadTimeout`,
  `FileNotFoundError`, `ImportError`
- `except Exception` only for fallback/cleanup paths
- `RuntimeError` for unrecoverable API failures
- Warnings to `sys.stderr`: `print("[WARN] ...", file=sys.stderr)`
- Fatal startup errors: `raise SystemExit(...)`
- No bare `except:` — always at minimum `except Exception`
- Silent `except Exception: pass` only in fire-and-forget cleanup

### Logging

No `logging` module — all output via `print()`. User-facing output to
`stdout`; warnings and errors to `stderr`. Do not introduce `logging` unless
explicitly requested.

---

## Profiles

Each user profile lives in `profiles/<name>/`. The directory may be empty —
missing files are silently tolerated (WARN printed to stderr, no crash).

`PROFILE.json` schema (required sections: `meta`, `style`, `constraints`,
`context`, `prompt_injection`):

```json
{
  "meta": { "id": "name", "description": "..." },
  "style": { "language": "ru", "verbosity": "concise", ... },
  "constraints": { "stack": {...}, "api": {...}, "architecture": {...} },
  "context": { "project": "...", "user": {...} },
  "prompt_injection": { "inject_as": "PROFILE", "enabled": true }
}
```

`to_yaml()` renders all sections except `meta` and `prompt_injection`.

---

## Constraints

- **No OpenAI SDK** — use `requests` for all API calls
- **Minimal dependencies** — add only when necessary; always pin versions
- **No system-level pip installs** — venv only
- **SSH-friendly** — no GUI, no browser, terminal output only
- **Token efficiency** — prefer smaller context windows
- Mixed Russian/English is intentional: code and identifiers are English;
  user-facing CLI output, prompt comments, and `system_prompt.txt` are in
  Russian — preserve this convention
