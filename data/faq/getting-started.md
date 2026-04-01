# Getting Started — AI Agent CLI

## Installation

### Prerequisites
- Python 3.12+
- Ubuntu 24.04 (or compatible Linux)
- OpenAI API key

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd ai_advent_challenge6

# Create virtual environment (required — no system-wide pip)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="sk-..."
```

### First Launch

```bash
python llm_agent_cli.py
```

The CLI starts in CHAT mode with the default profile. You will see:
```
Provider:      openai (gpt-4.1-mini)
Agents loaded: 6 agent(s): assistant, coder, planner, research, reviewer, support, validator
[CHAT:assistant] >
```

## Common Launch Options

| Flag | Description | Example |
|---|---|---|
| `--model` | Override LLM model | `--model gpt-4.1` |
| `--profile` | Use a named profile | `--profile alice` |
| `-t` | Set temperature (0.0-1.0) | `-t 0.7` |
| `--history-limit` | Max messages in context | `--history-limit 10` |
| `--provider ollama` | Use local Ollama models | `--provider ollama` |

## Essential Commands

| Command | What it does |
|---|---|
| `/help` | Show all available commands |
| `/help <question>` | Ask a question about the project (uses RAG + Git context) |
| `/exit` | Save state and exit |
| `/task <text>` | Set a task description |
| `/state PLAN` | Enter planning mode |
| `/state EXEC` | Enter execution mode (requires a plan) |
| `/step` | Execute next plan step (in EXEC mode) |
| `/reset` | Clear working memory and dialogue |
| `/agent list` | Show all available agents |
| `/agent <name>` | Use a specific agent for the next message |

## Workflow Example

```
[CHAT:assistant] > /task Implement a CSV parser
OK: task set

[CHAT:assistant] > /state PLAN
[PLAN:planner] > How should I implement the CSV parser?
... planner creates a step-by-step plan ...

[PLAN:planner] > /state EXEC
[EXEC:coder] > /step
... coder executes step 1 ...

[EXEC:coder] > /state VALI
... validator checks the result ...
```

## Profiles

Each profile lives in `profiles/<name>/` and contains:
- `PROFILE.json` — user preferences (language, verbosity, stack)
- `INVARIANTS.yaml` — project constraints (banned tools, architecture rules)
- `PROJECT_MEMORY.md` — long-term project context

The default profile is in `profiles/default/`.

## Getting Help

- `/help <question>` — ask the AI assistant about the project
- `/agent support` — switch to the support assistant for troubleshooting
- Check `docs/` directory for code style guides and best practices
