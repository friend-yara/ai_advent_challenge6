# AI Advent Challenge — Project Memory

Project purpose:
AI Advent Challenge.
Goal: systematic study of AI and LLMs through practical implementation.

Platform:
- Ubuntu 24.04 (VDS, SSH)
- Python 3.12
- Bash 5.2
- venv (.venv)

API:
- OpenAI Responses endpoint
- No SDK (requests only)

---

## Current Architecture (Week 2, Day 7)

Agent type:
- CLI-based single-agent loop
- REPL interface
- SSH-friendly

Main files:
- agent.py (core agent logic)
- llm_agent_cli.py (CLI loop)
- system_prompt.txt (external system prompt)
- state.toon (persistent state, TOON v3.0)
- pricing.json (manual pricing config)

---

## Agent Design

State model:
- stage: IDLE / PLAN / EXECUTE / REVIEW
- goal
- plan
- actions
- notes
- history (full message list)

Persistence:
- state stored in state.toon
- TOON Spec v3.0 via toons library
- Automatic load on startup
- Automatic save after each turn
- Automatic save on exit

Context control:
- System prompt narrowing
- Limited history window
- Explicit stage management

---

## Week 1 — LLM Fundamentals

- Built minimal OpenAI API client (requests, no SDK)
- Implemented temperature control
- Compared reasoning strategies
- Tested weak/medium/strong models
- Implemented token and cost tracking
- Added CLI argument handling

---

## Week 2 — Agents & Memory

- Implemented CLI agent
- Introduced single-agent loop architecture
- Added stage-based execution model
- Externalized system prompt
- Implemented TOON v3.0 state persistence
- Added automatic context restoration (Day 7)

---

## Current Capabilities

- Agent remembers conversation after restart
- Agent continues dialogue seamlessly
- Explicit state management
- Minimal dependencies
- PEP 668 compliant (venv only)

---

## Constraints

- No SDK
- Minimal architecture
- SSH-only usage
- Manual pricing updates
- Token efficiency preferred

---

## Next Focus

- Advanced context management
- Context pruning
- Structured memory layers
- Retrieval / indexing
- Agentic flow improvements
