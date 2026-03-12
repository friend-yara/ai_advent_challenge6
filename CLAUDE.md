# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Pure Python 3.12 CLI agent loop against the **OpenAI Responses API** (no SDK, no web framework). See `AGENTS.md` for the full authoritative reference — this file summarises the most important points.

## Environment

```bash
source .venv/bin/activate
export OPENAI_API_KEY="sk-..."
python llm_agent_cli.py            # start REPL (default profile)
python llm_agent_cli.py --profile alice -m gpt-4.1-mini
```

Dependencies are pinned in `requirements.txt` (`requests`, `toons`, `pyyaml`, `prompt_toolkit`). No build step, no linter, no test suite. If tests are added, use `pytest tests/`.

## Running the local MCP server

```bash
python mcp/server.py               # 127.0.0.1:8000
python mcp/server.py --port 9000
```

## Architecture overview

| File | Role |
|---|---|
| `llm_agent_cli.py` | CLI REPL — argparse, `/commands`, main loop |
| `orchestrator.py` | Routes messages to agents, enforces invariants |
| `agent.py` | Thin LLM caller (`_post` → OpenAI Responses API), memory persistence |
| `context_builder.py` | Assembles prompt blocks from live state layers |
| `agents.py` | `AgentSpec` dataclass + `AgentRegistry` (loads `agents/*.md`) |
| `mcp_client.py` | MCP Streamable HTTP client (pure `requests`, no MCP SDK) |
| `mcp/server.py` | Local MCP router — dispatches to domain modules |
| `mcp/base.py` | `MCPBaseHandler` — protocol mechanics shared across domain modules |

### Memory layers

Three independent layers, stored separately:

| Layer | Class | File | Contents |
|---|---|---|---|
| Short-term | `ShortTermMemory` | `short_term.toon` | messages, summary |
| Working | `TaskContext` | `state.toon` | task, state, plan, step, done |
| Long-term | `LongTermMemory` | YAML / JSON / MD | profile, invariants, project memory |

Both `.toon` files are gitignored and must never be committed. LTM files are **read-only at runtime**.

### State machine

`PLANNING → EXECUTION → VALIDATION → DONE`

Each state maps to a primary agent (`planner` / `coder` / `validator`). Agent auto-selection is state-based; `/agent <name>` pins an agent for one turn.

### Prompt block injection order

```
SYSTEM → PROFILE → PROJECT_MEMORY → INVARIANTS → STATE → RULES → VALIDATION → SUMMARY → FACTS → DIALOG
```

`RULES` injected only when `tc.current` is set (EXECUTION). `VALIDATION` injected only in VALIDATION state.

### MCP tool servers

External servers: `tools/*.yaml` (name + url). Local server: `mcp/server.py` with domain modules `mcp/mcp_weather.py` and `mcp/mcp_scheduler.py`. To add a new domain tool: create `mcp/mcp_<name>.py` with `<NAME>_TOOLS` list and `dispatch_<name>_tool()`, then register in `mcp/server.py` (`ALL_TOOLS` + `_DISPATCH`).

## Key constraints

- **No OpenAI SDK** — raw `requests` HTTP calls only
- **No system-wide pip installs** — venv only; always pin new deps in `requirements.txt`
- **No GUI** — SSH/terminal output only
- Mixed Russian/English is intentional: code/identifiers in English, user-facing output and `system_prompt.txt` in Russian

## Code style (brief)

- Python 3.10+ union types: `float | None`
- f-strings everywhere; `json.dumps(..., ensure_ascii=False, indent=2)`
- `print()` for all output — no `logging` module
- Warnings → `sys.stderr`; fatal startup errors → `raise SystemExit(...)`
- No bare `except:` — always `except Exception` at minimum
- Section separators: `# ---------------- Section ----------------`
