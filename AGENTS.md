# AGENTS.md — AI Advent Challenge

Guidance for agentic coding agents operating in this repository.

---

## Project Overview

An exploratory AI/LLM learning project (AI Advent Challenge). Pure Python CLI application
implementing an agent loop against the OpenAI Responses API — no SDK, no web framework.

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

Dependencies are fully pinned in `requirements.txt`. Do not install packages system-wide
(PEP 668 — venv only). After adding a dependency, pin its version in `requirements.txt`.

---

## Running the Project

| Command | Description |
|---|---|
| `python llm_agent_cli.py` | Start the agent REPL (main entry point) |
| `python llm_agent_cli.py -m gpt-4.1-mini` | Use a specific model |
| `python llm_agent_cli.py -t 0.7` | Set temperature |
| `python llm_agent_cli.py --history-limit 10` | Limit history window |
| `python llm_agent_cli.py --context-summary` | Enable context compression |
| `python llm_agent_cli.py -m gpt-4.1 -t 0.7 --history-limit 10 --context-summary` | Full options |
| `python 04_temperature/llm_request.py "Your prompt here"` | Day 4 standalone CLI |
| `python 04_temperature/llm_request.py -f path/to/prompt.txt` | Prompt from file |
| `python 01_simple_request/01_openai_request.py "Hello"` | Day 1 minimal script |

---

## Build / Lint / Test Commands

**There is no build step, no configured linter, and no test suite.**

- No `Makefile`, `pyproject.toml`, `tox.ini`, or `pytest.ini` exist.
- No test files exist. If you add tests, use `pytest` and place them in a `tests/` directory.
- If you add a linter (recommended: `ruff`), run it with `ruff check .`.
- If you add a formatter (recommended: `ruff format`), run it with `ruff format .`.

**Running a single test (if tests are added):**
```bash
pytest tests/test_agent.py::test_function_name -v
```

---

## Architecture

```
agent.py              Core Agent class — LLM calls, state, history, persistence
llm_agent_cli.py      CLI REPL entry point — argparse, REPL loop, /commands
system_prompt.txt     External system prompt (loaded at runtime)
state.toon            Persistent agent state (TOON v3.0, gitignored)
pricing.json          Manual pricing table for token cost calculation
requirements.txt      Pinned Python dependencies
PROJECT_MEMORY.md     Human-authored project context document
```

Weekly exercises live in numbered directories (`01_simple_request/`, `04_temperature/`, etc.).
Each is a self-contained script; do not refactor them into the core agent unless explicitly asked.

---

## API Usage

- **Endpoint:** `https://api.openai.com/v1/responses` (Responses API — not Chat Completions)
- **No OpenAI SDK** — all calls use `requests` with raw HTTP
- API key read from `OPENAI_API_KEY` environment variable
- Retry logic: 3 attempts with exponential backoff (2s → 4s → 8s) on `ReadTimeout`
- API errors detected via `data.get("error") is not None`, raised as `RuntimeError`

---

## Code Style

### Imports

Order: stdlib → third-party → local. One import per line. Use `from pathlib import Path`
for file operations in new code. Wrap optional imports in `try/except ImportError`.

```python
import json
import sys
import time
from pathlib import Path
from typing import Optional

import requests

from agent import Agent, load_pricing_models
```

Avoid comma-separated imports on one line (`import os, sys`) — that pattern appears only in
early sketch files and should not be replicated in new code.

### Type Hints

- Use Python 3.10+ union syntax: `float | None`, `list[str] | None` (not `Optional[float]`)
- Annotate all function parameters and return types for non-trivial functions
- Use plain `dict` for API payloads — no `TypedDict`, `dataclasses`, or `pydantic`
- `from typing import Optional` is present for legacy reasons; new code uses `X | None`

```python
def load_pricing_models(script_dir: Path | None = None) -> dict:
    ...

def reply(self, user_text: str) -> tuple[str, dict]:
    ...
```

### Naming Conventions

| Entity | Convention | Example |
|---|---|---|
| Variables, functions, methods | `snake_case` | `history_limit`, `load_pricing_models` |
| Classes | `PascalCase` | `Agent` |
| Module-level constants | `SCREAMING_SNAKE_CASE` | `URL = "https://..."` |
| Private methods | `_underscore_prefix` | `_build_prompt`, `_post` |
| CLI arguments | `--kebab-case` | `--max-output-tokens`, `--history-limit` |

### Formatting

- f-strings for all string interpolation
- `json.dumps(..., ensure_ascii=False, indent=2)` for JSON output (preserves Cyrillic)
- Section separators for logical groups inside long files:
  ```python
  # ---------------- Context management ----------------
  ```
- No line length enforcer configured; keep lines readable (aim for <100 chars)

### Docstrings

- Short one-line docstrings on all public methods and module-level functions
- Multi-line docstrings for classes, using a bullet list for capabilities
- Private methods may omit docstrings for trivial helpers

```python
def load_pricing_models(script_dir: Path | None = None) -> dict:
    """Load pricing models from pricing.json."""

class Agent:
    """
    Minimal LLM agent with:
    - history
    - stage-based state
    - TOON persistence
    """
```

### Error Handling

- Catch specific exceptions where possible: `requests.exceptions.ReadTimeout`, `FileNotFoundError`, `ImportError`
- Use broad `except Exception` only for fallback/cleanup paths (e.g., saving state on exit)
- Raise `RuntimeError` for unrecoverable API failures
- Print warnings to `sys.stderr`: `print("[WARN] ...", file=sys.stderr)`
- Use `raise SystemExit(...)` for fatal startup errors (missing dependencies, missing API key)
- Do not use bare `except:` — always at minimum `except Exception`
- Silent `except Exception: pass` is acceptable only in fire-and-forget cleanup (e.g., state save on REPL exit)

```python
# Good — specific exception
try:
    r = requests.post(URL, headers=headers, json=payload, timeout=self.timeout)
except requests.exceptions.ReadTimeout:
    if attempt == retries - 1:
        raise
    time.sleep(delay)
    delay *= 2

# Good — graceful optional dependency
try:
    import toons
except ImportError as e:
    raise SystemExit("ERROR: Install TOON support: pip install toons") from e

# Good — fallback with warning
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("[WARN] pricing.json not found, cost calculation disabled", file=sys.stderr)
    return {}
```

### Logging

No `logging` module is used — all output is via `print()`. User-facing output goes to
`stdout`; warnings and errors go to `stderr`. Do not introduce `logging` unless explicitly
requested.

---

## State and Persistence

- Agent state is stored in `state.toon` (TOON Spec v3.0 via `toons` library)
- State is loaded automatically on startup and saved after each REPL turn
- `state.toon` is gitignored — never commit it
- State schema: `stage` (IDLE/PLAN/EXECUTE/REVIEW), `goal`, `plan`, `actions`, `notes`, `history`

---

## Constraints

- **No OpenAI SDK** — use `requests` for all API calls
- **Minimal dependencies** — add packages only when necessary; always pin versions
- **No system-level pip installs** — venv only
- **SSH-friendly** — no GUI, no browser, terminal output only
- **Token efficiency** — prefer smaller context windows; use `--history-limit` and `--context-summary`
- Mixed Russian/English is intentional: code and identifiers are English; user-facing CLI output,
  comments in prompts, and `system_prompt.txt` are in Russian — preserve this convention
