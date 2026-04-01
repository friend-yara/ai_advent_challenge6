# Features — AI Agent CLI

## RAG Document Search

The CLI includes a RAG (Retrieval-Augmented Generation) system for searching project documentation.

### Indexing Documents

```bash
# Index a directory with fixed-size chunks (default)
python -m rag.indexer docs/ --chunker fixed

# Index with structured chunks (splits by Markdown headings)
python -m rag.indexer data/faq/ --chunker structured
```

### Using RAG in Conversations

When RAG is enabled, the system automatically searches indexed documents before answering. You can also use it explicitly:

- The LLM has access to `document_search` tool
- In "filter" mode (default for OpenAI), queries are rewritten for better retrieval
- In "base" mode (Ollama), direct similarity search is used

### RAG Quality Tips

- Use structured chunker for Markdown documents with clear headings
- Minimum 2 results needed for "strong context" (grounded answer)
- If results are poor, try reindexing with a different chunker strategy

## MCP Tools

The CLI connects to MCP (Model Context Protocol) servers for external tool access.

### Available Tools

| Server | Tools | Description |
|---|---|---|
| Weather | `get_forecast`, `summarize_forecast` | Weather forecasts by city |
| Scheduler | `reminder` | Delayed reminders with optional pipeline |
| Storage | `save_to_file`, `read_file` | File read/write operations |
| Git | `git_branch`, `git_log`, `git_diff`, `git_diff_branch` | Repository information |
| CRM | `get_ticket`, `search_tickets`, `list_user_tickets` | Support ticket management |

### Managing Tools

```
/tool list              — show all configured MCP servers
/tool list weather      — show tools from a specific server
```

MCP servers start automatically with the CLI. Configuration files are in `tools/*.yaml`.

### Adding a New MCP Server

1. Create `tools/<name>.yaml` with `name`, `url`, `description`
2. The server appears on next CLI startup

## Agent System

### Primary Agents (auto-selected by state)

| Agent | State | Model | Purpose |
|---|---|---|---|
| assistant | CHAT | gpt-4.1-mini | Direct Q&A, RAG, MCP tools |
| planner | PLANNING | gpt-4.1-mini | Decomposes tasks into plans |
| coder | EXECUTION | gpt-4.1 | Executes plan steps |
| validator | VALIDATION | gpt-4.1-mini | Validates completed work |
| support | CHAT | gpt-4.1-mini | User support with CRM + FAQ |

### Subagents (delegated by primary agents)

| Agent | Purpose |
|---|---|
| research | Factual/background research |
| reviewer | Code and work quality review |

Primary agents can automatically delegate tasks to subagents when needed. For example, planner may call `delegate_research` to get technical background before creating a plan.

### Manual Agent Selection

```
/agent support      — use support agent for next message
/agent research     — use research agent for next message
/agent list         — show all agents with their models and states
```

## Code Review

### Quick Review

```
/review HEAD~1              — review last commit
/review path/to/file.diff   — review from diff file
```

### GitHub PR Review

```
/review --pr 42 --repo owner/name --post
```

The review uses RAG context (code review checklist) and LLM analysis. With `--post`, the review is posted as a PR comment (requires GITHUB_TOKEN).

## Metrics and Quality

### Tracking Costs

Every LLM call shows token count and cost. Metrics are persisted to `metrics.jsonl`.

### Rating Answers

```
/rate 4             — rate the last answer (1-5 scale)
/metrics            — show aggregate statistics
```

Metrics include: total cost, average latency, average rating, pipeline success rate.

## Ollama (Local LLM)

Run models locally without API costs:

```bash
# Start with Ollama
python llm_agent_cli.py --provider ollama

# Specify model
python llm_agent_cli.py --provider ollama --ollama-model llama3.2
```

Ollama mode uses base RAG (no query rewriting) and skips invariant retries for performance.
