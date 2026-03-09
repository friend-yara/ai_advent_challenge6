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
| `llm_agent` | Same as above via symlink in ~/bin |

Flags `--use-project-memory` and `--use-invariants` default to `True`.
`--use-profile` is always `True` (not a CLI flag).
`--agents-dir` defaults to `agents/` (project root).
`--tools-dir` defaults to `tools/` (project root).

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
agent.py              Thin LLM caller — _post(), memory layers, persistence
agents.py             AgentSpec dataclass + AgentRegistry (loads agents/*.md)
context_builder.py    ContextBuilder — per-agent context assembly
orchestrator.py       Orchestrator — agent selection, routing, invariant checks
mcp_client.py         MCPClient — MCP Streamable HTTP client (pure requests)
llm_agent_cli.py      CLI REPL — argparse, REPL loop, /commands
system_prompt.txt     Fallback system prompt (used when no agent spec matched)
pricing.json          Manual pricing table for token cost calculation
requirements.txt      Pinned Python dependencies
tools/
  vkusvill.yaml       MCP server spec: ВкусВилл product search
agents/
  planner.md          Primary agent — PLANNING, model=gpt-4.1-mini
  coder.md            Primary agent — EXECUTION, model=gpt-4.1
  validator.md        Primary agent — VALIDATION, model=gpt-4.1-mini
  research.md         Subagent — PLANNING+EXECUTION, model=gpt-4.1-mini
  reviewer.md         Subagent — EXECUTION+VALIDATION, model=gpt-4.1-mini
profiles/
  default/
    PROFILE.json      User profile (JSON, rendered as YAML)
    INVARIANTS.yaml   Stack/arch/budget/banned constraints (YAML)
    PROJECT_MEMORY.md Project context document (plain text)
    state.toon        Working memory — gitignored
    short_term.toon   Dialogue history — gitignored
  <name>/             Additional user profiles (same structure)
```

---

## Memory Architecture

Three independent memory layers, each with its own storage:

| Layer | Class | File | Contents |
|---|---|---|---|
| Short-term | `ShortTermMemory` | `short_term.toon` | messages, summary |
| Working | `TaskContext` | `state.toon` | task, state, plan, step, done |
| Long-term | `LongTermMemory` | `.yaml` / `.md` / `.json` | profile, invariants, project memory |

**Rules:**
- `state.toon` must NEVER contain dialogue (`messages`/`summary`).
- `short_term.toon` must NEVER contain `TaskContext` fields.
- LTM files are read-only at runtime; never written by the agent.
- Both `.toon` files are gitignored — never commit them.

**Prompt block order:**
```
SYSTEM → PROFILE → PROJECT_MEMORY → INVARIANTS → STATE → RULES → VALIDATION → SUMMARY → FACTS → DIALOG
```

`RULES` block injected only when `tc.current` is set (EXECUTION state).
`VALIDATION` block injected only when `tc.state == "VALIDATION"` and
invariants are loaded.

---

## Key Classes in `agent.py`

### `Profile`
Loads `PROFILE.json`. Renders `style`/`constraints`/`context` as YAML via
`yaml.dump()` for prompt injection. `meta` and `prompt_injection` sections are
excluded from rendering. `prompt_injection.enabled` controls whether the block
is injected.

### `Invariants`
Loads `INVARIANTS.yaml`. `to_text()` renders human-readable structured text
for prompt injection (section headers + items, `meta` excluded).
`to_banned_lines()` returns the `banned:` list as `- item` lines for
`InvariantChecker`.

### `InvariantChecker`
Parses banned rules from `Invariants.to_banned_lines()`. Matches lines of the
form `- No <keyword>`. Each keyword maps to regex patterns
(`_KEYWORD_PATTERNS`). Two check points in `Orchestrator.reply()`:
- **Pre-check:** soft warn if user query matches banned patterns; LLM still
  called. Violations returned in `metrics["pre_violations"]`.
- **Post-check:** if LLM answer matches banned patterns:
  - In `PLANNING`: replace answer with correction message (no retry).
  - Other states: retry LLM once with correction prompt; if retry also fails,
    return a refusal string.

### `LongTermMemory`
Loads `Profile`, `PROJECT_MEMORY.md`, `INVARIANTS.yaml` into process cache.
Never persisted to disk. Holds `InvariantChecker` instance (`self.checker`).
`reload()` re-reads all files without restarting.

### `TaskContext`
Working memory state machine. States: `PLANNING → EXECUTION → VALIDATION →
DONE`. Strict transitions defined in `TASK_TRANSITIONS` constant.
`set_state()` validates the transition and returns an error string on failure
instead of raising. Fields: `task`, `state`, `step`, `total`, `plan`, `done`,
`current`, `actions`, `notes`.

### `ShortTermMemory`
Current dialogue only (`messages` list + `summary` string). Serialised to
`short_term.toon` separately from working state.

### `Agent`
Thin LLM caller. Does NOT build prompts or route messages — that is
`ContextBuilder` + `Orchestrator`'s job. Key methods:
- `_post(payload)` — HTTP call to OpenAI Responses API with retry.
- `welcome_back()` — LLM-generated resume message shown on startup.
- `whoami(profile_name)` — LLM-generated profile summary (≤80 words).
- `plan_from_reply(text)` — parses `[ ] step` / `1. step` lists from LLM reply.
- `format_todo()` — renders plan as `[x]/[ ]` checklist.
- `reset_working()` / `reset_short_term()` — layer-specific resets.
- `save_state` / `load_state` / `save_short_term` / `load_short_term` — TOON
  persistence (working and STM stored in separate files).
- `_compress_history_if_needed()` — compresses STM into summary when overflow.
- Backward-compat `@property` shims: `goal`, `plan`, `actions`, `notes`
  delegate to `self.tc`.

---

## Key Classes in `agents.py`

### `ContextPolicy`
Dataclass declaring which context layers an agent receives. Fields:
`include_profile`, `include_invariants`, `include_project_memory`,
`include_state`, `include_history`, `history_limit`, `include_rules_block`,
`include_validation_block`, `include_summary`, `include_facts`,
`include_task`, `include_plan_summary`.

### `AgentSpec`
Dataclass loaded from an `agents/*.md` file. Fields: `name`, `mode`
(`primary`|`subagent`), `description`, `model`, `temperature`, `when_to_use`,
`allowed_states`, `context_policy`, `prompt`.

### `AgentRegistry`
Loads and indexes `AgentSpec` objects from `agents/*.md`.
- `load(agents_dir)` — scans directory, parses YAML front-matter.
- `get(name)` — lookup by name.
- `for_state(state)` — returns first primary agent whose `allowed_states`
  contains the given state.
- `list_all()` / `list_primaries()` — enumeration.

---

## Key Classes in `context_builder.py`

### `ContextBuilder`
Assembles the LLM prompt string from live state layers, guided by an
`AgentSpec.context_policy`. Replaces the old `Agent._build_prompt`.

```
build(spec, user_text, tc, stm, ltm, facts) -> str
```

Block injection order (each conditional on `context_policy`):
```
SYSTEM → PROFILE → PROJECT_MEMORY → INVARIANTS → STATE → RULES
       → VALIDATION → SUMMARY → FACTS → DIALOG → Assistant:
```

---

## Key Classes in `orchestrator.py`

### `Orchestrator`
Routes user messages to specialized agents, composes context, calls LLM,
enforces invariants. Wraps `Agent` without modifying it.

- `reply(user_text, agent_name=None)` — auto-selects or uses pinned agent,
  builds context, calls LLM, runs pre/post invariant checks.
- `run_step()` — executes one EXECUTION step via the `coder` agent.
- `pin_agent(name)` — pins an agent for the next turn only.
- `current_agent_name()` — returns the name of the agent that would be
  selected right now.

**Auto-selection (state-based):**

| State | Primary agent | Model |
|---|---|---|
| PLANNING | planner | gpt-4.1-mini |
| EXECUTION | coder | gpt-4.1 |
| VALIDATION | validator | gpt-4.1-mini |
| DONE | fallback | (agent default) |

**Context per agent (what gets injected):**

| Agent | profile | invariants | project_mem | state | history | rules | validation |
|---|---|---|---|---|---|---|---|
| planner | ✓ | ✓ | ✓ | ✓ | 4 msgs | ✗ | ✗ |
| research | ✗ | ✗ | ✓ | task only | ✗ | ✗ | ✗ |
| coder | ✓ | ✓ | ✗ | ✓ | 4 msgs | ✓ | ✗ |
| reviewer | ✓ | ✓ | ✗ | plan+done | ✗ | ✗ | ✗ |
| validator | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ |

---

## Agent Spec File Format (`agents/*.md`)

Each file is a Markdown document with a YAML front-matter block. The text
after the closing `---` becomes the agent's system prompt.

```markdown
---
name: planner
mode: primary           # primary | subagent
description: ...
model: gpt-4.1-mini
temperature: 0.3
when_to_use: ...
allowed_states:
  - PLANNING
context_policy:
  include_profile: true
  include_invariants: true
  include_project_memory: true
  include_state: true
  include_history: true
  history_limit: 4
  include_rules_block: false
  include_validation_block: false
  include_summary: true
  include_facts: false
---

Ты — агент-планировщик. ...
```

---

## CLI Commands

| Command | Description |
|---|---|
| `/help` | Show command reference |
| `/exit` | Exit (state auto-saved) |
| `/reset` | Clear working memory + dialogue, save both files |
| `/save` | Save working state to `state.toon` |
| `/load` | Load working state from `state.toon` |
| `/task <text>` | Set task description in working memory |
| `/goal <text>` | Alias for `/task` |
| `/state <s>` | Transition state: `PLANNING`/`PLAN`, `EXECUTION`/`EXEC`, `VALIDATION`/`VALI`, `DONE` |
| `/step` | Execute next step (EXECUTION state only) |
| `/agent <name>` | Pin agent for next message only |
| `/agent list` | List all available agents with model and states |
| `/show` | Display working memory + STM + orchestrator status |
| `/system <text>` | Override system prompt for this session |
| `/checkpoint` | Save snapshot of current branch |
| `/branch list` | List all branches (`*` = active) |
| `/branch create <name>` | Create branch from current state |
| `/branch switch <name>` | Switch to branch, saving current state first |
| `/ltm reload` | Reload LTM files from disk without restarting |
| `/whoami` | LLM-generated summary of current profile (≤80 words) |
| `/tool list` | List all configured MCP servers |
| `/tool list <server>` | Connect to MCP server and print available tools |

**State prompt labels:** `[PLAN:planner] >`, `[EXEC:coder] >`, `[VALI:validator] >`, `[DONE:fallback] >`

**Auto-behaviours:**
- On startup with saved state: prints `welcome_back()` message and state hints.
- After each REPL turn: auto-saves `state.toon` and `short_term.toon`.
- `/state EXECUTION`: guarded — requires non-empty plan.
- `/state VALIDATION`: auto-runs `InvariantChecker` against the full plan.
- In PLANNING: if LLM reply contains a checklist, auto-populates `tc.plan`.

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

`INVARIANTS.yaml` schema (key sections: `stack`, `api`, `architecture`,
`budget`, `banned`). The `banned` list drives `InvariantChecker`:

```yaml
banned:
  - No OpenAI SDK
  - No system-wide pip installs
  - No GUI frameworks
```

---

## MCP Tool Servers

MCP server specs live in `tools/*.yaml`. Each file describes one server:

```yaml
name: vkusvill
url: https://mcp001.vkusvill.ru/mcp
description: ВкусВилл — поиск товаров и формирование ссылки на корзину
```

`MCPClient` in `mcp_client.py` implements the **MCP Streamable HTTP**
transport (spec version `2024-11-05`) using only `requests` — no MCP SDK,
no extra dependencies.

**Session lifecycle per call (stateless):**
1. `POST <url>` with `initialize` → capture `Mcp-Session-Id` response header
2. `POST <url>` with `notifications/initialized` (session header) → 202
3. `POST <url>` with `tools/list` (session header) → tools JSON array

**To add a new server:** create `tools/<name>.yaml` with `name` and `url`.
The server is immediately available as `/tool list <name>` on next startup
(or after reloading if hot-reload is added).

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
