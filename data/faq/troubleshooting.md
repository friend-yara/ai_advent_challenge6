# Troubleshooting — AI Agent CLI

## API and Authentication

### "OPENAI_API_KEY not set"

**Problem:** The CLI exits immediately with this error.

**Solution:**
```bash
export OPENAI_API_KEY="sk-..."
```

Make sure the key is valid. You can test it:
```bash
curl -s https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | head -c 200
```

If you don't want to use OpenAI, switch to Ollama:
```bash
python llm_agent_cli.py --provider ollama
```

### "API error: insufficient_quota"

**Problem:** Your OpenAI account has run out of credits.

**Solution:**
- Check your balance at https://platform.openai.com/usage
- Use a cheaper model: `--model gpt-4.1-mini` (10x cheaper than gpt-4.1)
- Switch to Ollama for free local inference: `--provider ollama`
- Check your spending with `/metrics` command

### "ReadTimeout" or slow responses

**Problem:** API calls are timing out.

**Solution:**
- The CLI retries 3 times with exponential backoff (2s → 4s → 8s)
- If persistent, check your internet connection
- Try a faster model: `--model gpt-4.1-mini`
- Increase timeout: `--timeout 60`

## MCP Tools

### "MCP tools not available" or "Tool not found"

**Problem:** MCP servers didn't start or aren't responding.

**Solution:**
1. Check if servers are running: `curl http://127.0.0.1:8001/mcp` (should respond)
2. MCP servers auto-start with the CLI. If they failed:
   ```bash
   # Start manually in a separate terminal
   python mcp/server.py
   ```
3. Check `tools/*.yaml` files — each must have `name` and `url`
4. Use `/tool list` to see configured servers

### "Tool call failed" in conversation

**Problem:** The LLM called a tool but it returned an error.

**Solution:**
- Check the error message in stderr (it shows the tool name and error)
- Weather tool: the city name must be in English
- Git tools: make sure you're in a git repository
- Storage tools: files are restricted to the project root directory

### Sandbox blocks a tool

**Problem:** You see `[SANDBOX] Опасный инструмент: ...` and the tool doesn't execute.

**Explanation:** Some tools are classified as "dangerous" (save_to_file, reminder). The CLI asks for confirmation before executing them.

**Solution:** Type `y` or `yes` to allow, `n` to deny. The LLM will handle the denial gracefully.

## RAG and Document Search

### "RAG index not found"

**Problem:** Document search returns no results or shows a warning.

**Solution:**
```bash
# Index your documents
source .venv/bin/activate
python -m rag.indexer docs/ --chunker fixed

# For FAQ documents with clear headings
python -m rag.indexer data/faq/ --chunker structured
```

### RAG returns irrelevant results

**Problem:** Search results don't match the query.

**Possible causes:**
1. **Wrong language:** RAG index language differs from query language. The system warns: `[RAG WARN] Язык запроса 'ru' отличается от языка индекса 'en'`
2. **Threshold too high:** Try lowering the similarity threshold
3. **Wrong chunker:** Structured chunker works better for Markdown with headings
4. **Index stale:** Reindex after adding new documents

### "No context available" in /help

**Problem:** `/help <question>` says no RAG context.

**Solution:** Index the project documents first:
```bash
python -m rag.indexer docs/ --chunker fixed
```

## State Machine

### "Plan is empty — create a plan first"

**Problem:** Trying to enter EXECUTION state without a plan.

**Solution:**
1. Set a task: `/task <description>`
2. Enter planning: `/state PLAN`
3. Ask the planner to create a plan
4. The plan auto-populates from the LLM response
5. Then: `/state EXEC`

### State transitions don't work

**Valid transitions:**
```
CHAT → PLANNING → EXECUTION → VALIDATION → DONE
                ↑                          ↓
                └──────────────────────────┘
```

You cannot skip states (e.g., CHAT → EXECUTION). Follow the sequence.

### Working memory is corrupted

**Problem:** Strange errors about state.toon or short_term.toon.

**Solution:**
```bash
# Reset everything
/reset

# Or manually delete state files
rm profiles/default/state.toon profiles/default/short_term.toon
```

## Cost Management

### How to reduce API costs

1. **Use gpt-4.1-mini** instead of gpt-4.1 (10x cheaper): `--model gpt-4.1-mini`
2. **Use Ollama** for free local inference: `--provider ollama`
3. **Limit history** to reduce input tokens: `--history-limit 4`
4. **Monitor spending** with `/metrics` command
5. **Check per-request costs** — shown after each response

### Understanding the cost display

After each response you see:
```
(openai/gpt-4.1-mini, in=450, out=120, cost=$0.000180)
```
- `in=450` — input tokens (your message + context)
- `out=120` — output tokens (LLM response)
- `cost=$0.000180` — total cost for this request
