# Project Invariants

## Stack
- Language: Python 3.12
- Platform: Ubuntu 24.04, SSH-only
- Shell: Bash 5.2
- Environment: venv (.venv), PEP 668 compliant — no system-level pip installs

## API
- Endpoint: OpenAI Responses API (https://api.openai.com/v1/responses)
- No OpenAI SDK — raw HTTP via requests only
- API key from OPENAI_API_KEY environment variable

## Architecture Constraints
- Single-file core: agent.py + llm_agent_cli.py
- No web framework, no GUI, no browser
- Minimal dependencies — add only when necessary, always pin versions
- No database — file-based persistence only (TOON format)

## Budget
- Prefer smaller models (gpt-4.1-mini) for exploration tasks
- Use gpt-4.1 only for complex reasoning or generation tasks
- Token efficiency is a priority; avoid redundant context

## Banned
- No SDK wrappers (openai, langchain, etc.)
- No system-wide pip installs
- No GUI or browser-based tools
- No cloud storage or external APIs beyond OpenAI
