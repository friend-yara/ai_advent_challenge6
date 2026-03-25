#!/usr/bin/env python3
"""
orchestrator.py — Orchestrator.

Routes user messages to specialized agents based on the current task state,
builds context precisely per agent spec, calls the LLM, enforces invariants,
and executes MCP tool calls when the LLM requests them.

Demo flow:
  [PLAN] > Реализуй функцию parse_csv на Python
      → planner selected (state=PLANNING, model=gpt-4.1-mini)
      → context: profile + invariants + project_memory + state + history(4)

  /state EXEC
  [EXEC] > /step
      → coder selected (state=EXECUTION, model=gpt-4.1)
      → context: profile + invariants + state + history(4) + RULES

  /state VALI
      → validator selected (state=VALIDATION, model=gpt-4.1-mini)
      → context: invariants + state + VALIDATION block (no profile, no history)

  [PLAN] > Какая погода в Лондоне завтра?
      → planner selected
      → LLM call 1: requests tool get_forecast(place=London, days=1)
      → MCPClient.call_tool("weather", "get_forecast", {...})
      → LLM call 2: formulates final answer with weather data
      → answer ends with  #weather
"""

import json
import sys

from agents import AgentRegistry, AgentSpec
from context_builder import ContextBuilder

try:
    from langdetect import detect as _langdetect
    from langdetect.lang_detect_exception import LangDetectException as _LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

# ---------------- RAG tool definition ----------------

_DOCUMENT_SEARCH_TOOL = {
    "type": "function",
    "name": "document_search",
    "description": (
        "Search the local document corpus for chunks relevant to a query. "
        "Use this to retrieve context from indexed project documents."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant document chunks.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top results to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


def rag_available() -> bool:
    """Return True if the RAG index exists and rag package is importable."""
    try:
        from pathlib import Path
        from rag.retriever import _DEFAULT_INDEX_DIR
        config = _DEFAULT_INDEX_DIR / "config.json"
        return config.exists()
    except Exception:
        return False


class Orchestrator:
    """
    Selects the appropriate agent for the current task state,
    composes a precise context, calls the LLM, enforces invariants,
    and executes MCP tool calls when the LLM requests them.

    Wraps Agent (for _post, state, memory) without modifying it.
    """

    def __init__(
        self,
        agent,                  # Agent instance (thin caller + state)
        registry: AgentRegistry,
        context_builder: ContextBuilder,
        pricing: dict,
        mcp=None,               # MCPClient | None
    ):
        """Initialize orchestrator with all collaborators."""
        self.agent = agent
        self.registry = registry
        self.ctx = context_builder
        self.pricing = pricing
        self.mcp = mcp
        # Pinned agent for the next turn only (set via /agent <name>)
        self._pinned_agent: str | None = None
        # RAG mode — "off" | "base" | "filter"; set at REPL startup if index exists
        self.rag_mode: str = "off"
        self.rag_initial_top_k: int = 15
        self.rag_final_top_k: int = 5
        self.rag_similarity_threshold: float = 0.38
        self.rag_use_keyword_filter: bool = False
        self.rag_index_lang: str = "en"  # ожидаемый язык индекса
        self.rag_min_results: int = 1    # минимум чанков для "strong context"
        self.last_rag_results: list = []
        self.last_rag_metadata: dict = {}
        self.last_rag_lang_warn: str | None = None
        self.rag_task_memory: dict = {"goal": "", "key_terms": [], "turn_count": 0}

    @property
    def rag_enabled(self) -> bool:
        """Derived: True when rag_mode is 'base' or 'filter'."""
        return self.rag_mode != "off"

    def _detect_lang(self, text: str) -> str | None:
        """Detect language of text; returns ISO 639-1 code or None on failure."""
        if not _LANGDETECT_AVAILABLE:
            return None
        try:
            return _langdetect(text)
        except _LangDetectException:
            return None

    # ---------------- Public API ----------------

    def reply(self, user_text: str, agent_name: str | None = None
              ) -> tuple[str, dict]:
        """
        Process a user message: select agent → build context → call LLM.
        If the LLM requests tool calls and MCP is configured, execute them
        and make a second LLM call with the results injected.

        Parameters
        ----------
        user_text  : message from the user
        agent_name : explicit agent name override; None = auto-select by state

        Returns
        -------
        (reply_text, metrics_dict)
        metrics includes: model, time, in, out, cost, agent, pre_violations
        """
        self.last_rag_lang_warn = None
        if self.rag_enabled:
            lang = self._detect_lang(user_text)
            if lang and lang != self.rag_index_lang:
                self.last_rag_lang_warn = (
                    f"[RAG WARN] Язык запроса '{lang}' отличается от языка индекса"
                    f" '{self.rag_index_lang}'. Качество поиска может быть низким."
                )

        ag = self.agent
        tc = ag.tc
        stm = ag.stm
        ltm = ag.ltm

        # Select agent spec
        name = agent_name or self._pinned_agent
        self._pinned_agent = None   # consume pin after one use
        spec = self._select_spec(name, tc.state)

        # Record message in STM
        stm.messages.append({"role": "user", "text": user_text})

        # Facts extraction (when strategy == "facts")
        if ag.context_strategy == "facts":
            _update_facts(ag.facts, user_text)

        # Summary compression before building prompt
        ag._compress_history_if_needed()

        # Pre-check invariants (soft warn)
        pre_violations: list[str] = []
        if ltm.invariants and ltm.checker.rules:
            _, pre_violations = ltm.checker.check(user_text)

        # Build tools list for LLM (planner only, if MCP is available)
        tools_for_llm = self._build_tools_list(spec)

        # Build context prompt (skip heavy context for local providers)
        if ag.provider.name == "ollama":
            prompt = user_text
        else:
            prompt = self.ctx.build(spec, user_text, tc, stm, ltm, ag.facts)

        # RAG pre-search: always search before LLM call, inject results into prompt
        if self.rag_enabled:
            is_local = ag.provider.name == "ollama"
            rag_query = user_text if is_local else self._build_rag_query(user_text)
            rag_top_k = 3 if is_local else 5
            rag_text, _, rag_data = self._execute_document_search(
                {"query": rag_query, "top_k": rag_top_k}
            )
            if is_local:
                # Compact prompt: grounding instruction + context + question
                prompt = (
                    "Answer ONLY from the documents below. "
                    "If the documents don't contain enough information, say so.\n\n"
                    + rag_text + "\n\nQuestion: " + user_text
                )
            else:
                # OpenAI path: inject task memory + RAG into built context
                tm = self.rag_task_memory
                if tm["goal"]:
                    task_ctx = (
                        f"\n\nRAG_CONTEXT:\n"
                        f"Research goal: {tm['goal']}\n"
                        f"Key terms: {', '.join(tm['key_terms'])}"
                    )
                    prompt += task_ctx
                prompt += "\n\n" + rag_text
            # Remove tool from list — search already executed by the system
            tools_for_llm = [t for t in tools_for_llm if t["name"] != "document_search"]

        # ---------- LLM call 1 ----------
        payload: dict = {"model": spec.model, "input": prompt}
        if spec.temperature is not None:
            payload["temperature"] = spec.temperature
        if ag.max_output_tokens is not None:
            payload["max_output_tokens"] = ag.max_output_tokens
        if ag.stop:
            payload["stop"] = ag.stop
        # Strip tools for providers that don't support them
        if not ag.provider.supports_tools:
            tools_for_llm = []

        if tools_for_llm:
            payload["tools"] = tools_for_llm

        data, elapsed = ag._post(payload)

        if isinstance(data, dict) and data.get("error") is not None:
            err = data.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(msg or "API error")

        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected API response type: {type(data)}")

        # Detect tool calls in the response
        output_items: list[dict] = data.get("output", [])
        tool_call_items = [
            item for item in output_items
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]

        tools_used: list[str] = []         # tool names used (for tag)
        tool_results_data: list[dict] = []  # structured results for CLI poller

        _MAX_TOOL_ROUNDS = 8
        all_tool_results: list[tuple[dict, str]] = []

        _round = 0
        while tool_call_items and _round < _MAX_TOOL_ROUNDS:
            _round += 1
            for tc_item in tool_call_items:
                result_text, server_name, result_data = self._execute_tool_call(tc_item)
                all_tool_results.append((tc_item, result_text))
                tc_tool_name = tc_item.get("name", "")
                if tc_tool_name and tc_tool_name not in tools_used:
                    tools_used.append(tc_tool_name)
                # Parse arguments for CLI consumers (e.g. reminder poller)
                raw_args = tc_item.get("arguments", "{}")
                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    arguments = {}
                tool_results_data.append({
                    "tool_name": tc_tool_name,
                    "server": server_name,
                    "arguments": arguments,
                    "result_text": result_text,
                    "data": result_data,
                })

            # ---------- LLM call with accumulated tool results ----------
            data_next, elapsed_next = self._call_with_tool_results(
                spec, user_text, all_tool_results, tools_for_llm, ag
            )
            elapsed += elapsed_next

            if isinstance(data_next, dict) and data_next.get("error") is not None:
                err = data_next.get("error") or {}
                msg = err.get("message") if isinstance(err, dict) else str(err)
                raise RuntimeError(msg or "API error (tool result call)")

            data = data_next
            output_items = data.get("output", []) if isinstance(data, dict) else []
            tool_call_items = [
                item for item in output_items
                if isinstance(item, dict) and item.get("type") == "function_call"
            ]

        # Extract final text
        if ag.print_json:
            text = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            text = _extract_text(data)

        # Post-check invariants (skip retry for slow local providers)
        if ltm.invariants and ltm.checker.rules:
            post_passed, post_violations = ltm.checker.check(text)
            if not post_passed:
                if tc.state == "PLANNING":
                    text = _planning_violation_message(post_violations)
                elif ag.provider.name != "ollama":
                    text = self._retry_with_violations(text, post_violations, spec)
                else:
                    vlist = "\n".join(f"  - {v}" for v in post_violations)
                    print(f"[WARN] Инварианты нарушены (retry пропущен для Ollama):\n{vlist}",
                          file=sys.stderr)

        # Append tool tag(s) if MCP tools were used (tool name, not server name)
        if tools_used:
            tag = " ".join(f"#{t}" for t in tools_used)
            text = f"{text} {tag}"

        stm.messages.append({"role": "assistant", "text": text})

        # Update RAG task memory after each RAG turn (skip for Ollama — requires OpenAI)
        if self.rag_enabled and ag.provider.name != "ollama":
            self._update_rag_task_memory(user_text, text)

        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        actual_model = ag.provider.resolve_model(spec.model)
        _, _, total_cost = _compute_cost(self.pricing, actual_model, in_tok, out_tok)

        metrics = {
            "provider": ag.provider.name,
            "model": actual_model,
            "agent": spec.name,
            "time": elapsed,
            "in": in_tok,
            "out": out_tok,
            "cost": total_cost,
            "pre_violations": pre_violations,
            "tool_results": tool_results_data,  # for CLI consumers (reminder poller etc.)
        }
        return text, metrics

    def run_step(self) -> tuple[str, dict]:
        """
        Execute the current EXECUTION step via the coder agent.

        Returns (reply_text, metrics_dict).
        Advances tc.step, tc.done, tc.current.
        """
        ag = self.agent
        tc = ag.tc
        stm = ag.stm
        ltm = ag.ltm

        if tc.step >= tc.total:
            return "Все шаги уже выполнены.", _zero_metrics(ag.model, ag.provider.name)

        current_step = tc.plan[tc.step]
        step_prompt = (
            f"Выполни шаг {tc.step + 1} из {tc.total}: {current_step}\n"
            f"Задача: {tc.task}\n"
            "Выполни только этот шаг. Не переходи к следующим."
        )

        # Use coder spec for execution steps; fall back to state-based default
        spec = self.registry.get("coder") or self._select_spec(None, tc.state)

        stm.messages.append({"role": "user", "text": step_prompt})
        ag._compress_history_if_needed()

        prompt = self.ctx.build(spec, step_prompt, tc, stm, ltm, ag.facts)

        payload: dict = {"model": spec.model, "input": prompt}
        if spec.temperature is not None:
            payload["temperature"] = spec.temperature
        if ag.max_output_tokens is not None:
            payload["max_output_tokens"] = ag.max_output_tokens

        data, elapsed = ag._post(payload)
        if isinstance(data, dict) and data.get("error"):
            err = data.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(msg or "API error (run_step)")

        text = _extract_text(data)
        stm.messages.append({"role": "assistant", "text": text})

        # Advance task state
        tc.done.append(current_step)
        tc.step += 1
        tc.current = tc.plan[tc.step] if tc.step < tc.total else ""

        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        actual_model = ag.provider.resolve_model(spec.model)
        _, _, total_cost = _compute_cost(self.pricing, actual_model, in_tok, out_tok)

        metrics = {
            "provider": ag.provider.name,
            "model": actual_model,
            "agent": spec.name,
            "time": elapsed,
            "in": in_tok,
            "out": out_tok,
            "cost": total_cost,
            "pre_violations": [],
        }
        return text, metrics

    def pin_agent(self, name: str) -> str | None:
        """
        Pin an agent by name for the next turn only.
        Returns error string if agent not found, else None.
        """
        if self.registry.get(name) is None:
            return f"Unknown agent '{name}'. Available: {self._agent_names()}"
        self._pinned_agent = name
        return None

    def current_agent_name(self) -> str:
        """Return the name of the agent that would be selected right now."""
        name = self._pinned_agent
        if name:
            return name
        spec = self.registry.for_state(self.agent.tc.state)
        return spec.name if spec else "(none)"

    # ---------------- Pipeline / routing ----------------

    def _route_tool(self, tool_name: str) -> str | None:
        """Return the MCP server name that provides the named tool, or None."""
        return self.mcp.find_tool_server(tool_name) if self.mcp else None

    def _build_capability_registry(self) -> dict:
        """Build {tool_name: server_name} from cached MCP tools."""
        if not self.mcp:
            return {}
        registry: dict[str, str] = {}
        for server_name, tools in self.mcp._tools_cache.items():
            for t in tools:
                registry[t["name"]] = server_name
        return registry

    def execute_pipeline(
        self,
        pipeline: list[dict],
        job_id: str,
        progress_cb=None,
    ) -> None:
        """
        Execute pipeline steps sequentially, chaining prev_output between steps.

        progress_cb — optional callable(message: str) for CLI output.
        Strips 'functions.' prefix from tool names if present (OpenAI quirk).
        """
        prev_output: str = ""
        last_idx = len(pipeline) - 1

        def _emit(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)

        for i, step in enumerate(pipeline):
            tool_name = step.get("tool", "")
            # Normalize: LLM sometimes prefixes names with "functions."
            if tool_name.startswith("functions."):
                tool_name = tool_name[len("functions."):]
            args = step.get("args", {})

            # Substitute {prev_output} in string arg values
            resolved_args = {
                k: (v.replace("{prev_output}", prev_output) if isinstance(v, str) else v)
                for k, v in args.items()
            }

            server_name = self._route_tool(tool_name)
            if server_name is None:
                _emit(f"\n[WARN] {tool_name}: not found, skipping")
                continue

            # Progress message before calling
            if tool_name == "get_forecast":
                status_msg = f"получаю прогноз (job #{job_id})"
            elif tool_name == "summarize_forecast":
                status_msg = f"составляю сводку (job #{job_id})"
            elif tool_name == "save_to_file":
                fname = resolved_args.get("filename", "forecast.txt")
                status_msg = f"{fname} (job #{job_id})"
            else:
                status_msg = f"(job #{job_id})"

            suffix = " - пайплайн закончен!" if i == last_idx else ""
            _emit(f"\n{server_name}:{tool_name}: {status_msg}{suffix}")

            try:
                result = self.mcp.call_tool(server_name, tool_name, resolved_args)
                content = result.get("content", [])
                prev_output = content[0].get("text", "") if content else ""
            except Exception as e:
                print(f"[WARN] Pipeline step '{tool_name}' failed: {e}", file=sys.stderr)
                _emit(f"[ERROR] {tool_name}: {e} — pipeline прерван")
                break

    # ---------------- MCP tool calling ----------------

    def _build_tools_list(self, spec: AgentSpec) -> list[dict]:
        """
        Return OpenAI function definitions for tools available to this agent.
        MCP tools: planner only (unchanged).
        document_search (RAG): all agents, when self.rag_enabled is True.
        Internal '_mcp_server' keys are stripped before sending to the API.
        """
        tools: list[dict] = []

        # MCP tools: planner and assistant
        if spec.name in ("planner", "assistant") and self.mcp is not None:
            raw = self.mcp.all_tools_for_llm()
            if raw:
                tools.extend(
                    {k: v for k, v in tool.items() if not k.startswith("_")}
                    for tool in raw
                )

        # RAG tool: all agents when enabled
        if self.rag_mode != "off":
            tools.append(_DOCUMENT_SEARCH_TOOL)

        return tools

    def _execute_tool_call(self, tc_item: dict) -> tuple[str, str | None, dict]:
        """
        Execute a single function_call item from the LLM output.

        Handles document_search locally; delegates everything else to MCPClient.
        Returns (result_text, server_name, result_data).
        On any error returns (error_description, None, {}).
        """
        tool_name = tc_item.get("name", "")
        raw_args = tc_item.get("arguments", "{}")
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            arguments = {}

        # Handle local RAG document search
        if tool_name == "document_search":
            return self._execute_document_search(arguments)

        if self.mcp is None:
            return (
                f"Tool '{tool_name}' is not available: MCP not configured.",
                None,
                {},
            )

        server_name = self.mcp.find_tool_server(tool_name)
        if server_name is None:
            return (
                f"Tool '{tool_name}' is not available on any configured MCP server.",
                None,
                {},
            )

        try:
            result_dict = self.mcp.call_tool(server_name, tool_name, arguments)
            # Extract text for LLM injection (function_call_output must be a string)
            content = result_dict.get("content", [])
            if content and isinstance(content, list):
                result_text = content[0].get("text", str(result_dict))
            else:
                result_text = str(result_dict.get("data", result_dict))
            result_data = result_dict.get("data", {})
            return result_text, server_name, result_data
        except Exception as e:
            print(f"[WARN] Tool call '{tool_name}' failed: {e}", file=sys.stderr)
            return f"Tool call failed: {e}", None, {}

    def _rewrite_query_for_retrieval(self, query: str) -> str:
        """Rewrite user query into retrieval-optimised keywords via LLM mini-call.

        On any exception falls back to returning the original query unchanged.
        """
        prompt = (
            "Convert the question into English keywords for document search. "
            "Output only keywords, no explanations.\n"
            f"Question: {query}"
        )
        payload: dict = {
            "model": "gpt-4.1-mini",
            "input": prompt,
            "max_output_tokens": 60,
            "temperature": 0.0,
        }
        try:
            data, _ = self.agent._post(payload)
            rewritten = _extract_text(data).strip()
            return rewritten if rewritten else query
        except Exception:
            return query

    def _build_rag_query(self, user_text: str) -> str:
        """Augment user query with accumulated key_terms from task memory."""
        terms = self.rag_task_memory.get("key_terms", [])
        if not terms:
            return user_text
        terms_str = " ".join(terms)
        return f"{user_text} {terms_str}"

    def _update_rag_task_memory(self, user_text: str, assistant_text: str) -> None:
        """Extract goal + key terms from latest turn via LLM mini-call.

        Merges new terms with existing (deduplicated, capped at 15).
        On any exception silently falls back (no crash).
        """
        current = self.rag_task_memory
        prompt = (
            "You are a research assistant tracker. Given the conversation turn below "
            "and the current research state, extract:\n"
            "1) The user's research goal (one sentence)\n"
            "2) New key terms/entities from this turn\n\n"
            f"Current goal: {current['goal'] or '(none yet)'}\n"
            f"Current terms: {', '.join(current['key_terms']) or '(none)'}\n\n"
            f"User: {user_text}\n"
            f"Assistant: {assistant_text[:500]}\n\n"
            "Output exactly two lines:\n"
            "goal: <one sentence>\n"
            "terms: <comma-separated new terms only>"
        )
        payload: dict = {
            "model": "gpt-4.1-mini",
            "input": prompt,
            "max_output_tokens": 80,
            "temperature": 0.0,
        }
        try:
            data, _ = self.agent._post(payload)
            reply = _extract_text(data).strip()
            for line in reply.splitlines():
                low = line.strip().lower()
                if low.startswith("goal:"):
                    current["goal"] = line.strip()[5:].strip()
                elif low.startswith("terms:"):
                    raw = line.strip()[6:].strip()
                    new_terms = [t.strip() for t in raw.split(",") if t.strip()]
                    existing = set(t.lower() for t in current["key_terms"])
                    for t in new_terms:
                        if t.lower() not in existing:
                            current["key_terms"].append(t)
                            existing.add(t.lower())
                    current["key_terms"] = current["key_terms"][:15]
            current["turn_count"] += 1
        except Exception:
            current["turn_count"] += 1

    def _execute_document_search(self, arguments: dict) -> tuple[str, str, dict]:
        """
        Execute a local RAG document_search tool call.

        Returns (result_text, server_name, result_data).
        Behaviour depends on self.rag_mode: "base" uses simple search; "filter"
        uses query rewriting + threshold filtering.

        Grounding logic:
        - weak context (0 results): tool instructs LLM to refuse answering.
        - strong context (results present): tool instructs LLM to answer ONLY
          from retrieved chunks; REPL handles Sources/Quotes rendering.
        """
        query = arguments.get("query", "")
        top_k = int(arguments.get("top_k", 5))
        _GROUNDED_HINT = (
            "\n\n---\n"
            "Ответь на вопрос, опираясь ТОЛЬКО на приведённые документы. "
            "Не используй внешние знания."
        )
        _WEAK_CONTEXT_MSG = (
            "Контекст: релевантных документов не найдено.\n\n"
            "ВАЖНО: Ты НЕ ДОЛЖЕН отвечать по существу вопроса. "
            "Скажи пользователю, что не удалось найти релевантную информацию, "
            "и попроси уточнить запрос. Не используй внешние знания."
        )
        try:
            if self.rag_mode == "base":
                from rag.retriever import search
                results = search(query, top_k=top_k)
                result_data = {"results": results, "mode": "base"}
                if len(results) < self.rag_min_results:
                    self.last_rag_results = []
                    self.last_rag_metadata = result_data
                    return _WEAK_CONTEXT_MSG, "rag", result_data
                self.last_rag_results = results
                self.last_rag_metadata = result_data
                lines = []
                for i, r in enumerate(results, 1):
                    section = f" [{r['section']}]" if r["section"] else ""
                    lines.append(
                        f"[{i}] {r['source']}{section} (score={r['score']:.3f})\n{r['text']}"
                    )
                result_text = "\n\n".join(lines) + _GROUNDED_HINT
                return result_text, "rag", result_data
            else:
                # rag_mode == "filter"
                from rag.retriever import search_improved
                rewritten = self._rewrite_query_for_retrieval(query)
                fallback_used = False
                final_results, dropped = search_improved(
                    rewritten,
                    initial_top_k=self.rag_initial_top_k,
                    final_top_k=self.rag_final_top_k,
                    threshold=self.rag_similarity_threshold,
                    use_keyword=self.rag_use_keyword_filter,
                )
                # Fallback: if rewritten query returned too few results,
                # retry with the original query
                if len(final_results) < self.rag_min_results and rewritten != query:
                    final_results, dropped = search_improved(
                        query,
                        initial_top_k=self.rag_initial_top_k,
                        final_top_k=self.rag_final_top_k,
                        threshold=self.rag_similarity_threshold,
                        use_keyword=self.rag_use_keyword_filter,
                    )
                    fallback_used = True
                result_data = {
                    "results": final_results,
                    "mode": "filter",
                    "original_query": query,
                    "rewritten_query": rewritten if rewritten != query else None,
                    "fallback_used": fallback_used,
                    "initial_count": self.rag_initial_top_k,
                    "filtered_out": dropped,
                }
                if len(final_results) < self.rag_min_results:
                    self.last_rag_results = []
                    self.last_rag_metadata = result_data
                    return _WEAK_CONTEXT_MSG, "rag", result_data
                self.last_rag_results = final_results
                self.last_rag_metadata = result_data
                lines = []
                for i, r in enumerate(final_results, 1):
                    section = f" [{r['section']}]" if r["section"] else ""
                    lines.append(
                        f"[{i}] {r['source']}{section} (score={r['score']:.3f})\n{r['text']}"
                    )
                result_text = "\n\n".join(lines) + _GROUNDED_HINT
                return result_text, "rag", result_data
        except Exception as e:
            print(f"[WARN] document_search failed: {e}", file=sys.stderr)
            self.last_rag_results = []
            self.last_rag_metadata = {}
            return f"document_search failed: {e}", "rag", {}

    def _call_with_tool_results(
        self,
        spec: AgentSpec,
        user_text: str,
        tool_results: list[tuple[dict, str]],
        tools_for_llm: list[dict],
        ag,
    ) -> tuple[dict, float]:
        """
        Make a second LLM call with tool results injected as input items.

        The input is an array with:
          - the original user message
          - one function_call item per tool call
          - one function_call_output item per tool result
        """
        input_items: list[dict] = [{"role": "user", "content": user_text}]

        for tc_item, result_text in tool_results:
            # Replay the function_call the LLM made
            input_items.append({
                "type": "function_call",
                "name": tc_item["name"],
                "call_id": tc_item["call_id"],
                "arguments": tc_item.get("arguments", "{}"),
            })
            # Inject our tool execution result
            input_items.append({
                "type": "function_call_output",
                "call_id": tc_item["call_id"],
                "output": result_text,
            })

        payload: dict = {
            "model": spec.model,
            "input": input_items,
        }
        if tools_for_llm:
            payload["tools"] = tools_for_llm
        if spec.temperature is not None:
            payload["temperature"] = spec.temperature
        if ag.max_output_tokens is not None:
            payload["max_output_tokens"] = ag.max_output_tokens

        return ag._post(payload)

    # ---------------- Internals ----------------

    def _select_spec(self, name: str | None, state: str) -> AgentSpec:
        """
        Select agent spec by name or auto-select by state.
        Falls back to a minimal inline spec if no agent is found.
        """
        if name:
            spec = self.registry.get(name)
            if spec:
                return spec
            print(
                f"[WARN] Agent '{name}' not found, falling back to "
                f"state-based selection",
                file=sys.stderr,
            )

        spec = self.registry.for_state(state)
        if spec:
            return spec

        print(
            f"[WARN] No primary agent for state '{state}', using default model",
            file=sys.stderr,
        )
        return _fallback_spec(self.agent.model, self.agent.system_prompt)

    def _retry_with_violations(
        self, answer: str, violations: list[str], spec: AgentSpec
    ) -> str:
        """Retry LLM once with a correction prompt. Return refusal on second failure."""
        violation_list = "\n".join(f"  - {v}" for v in violations)
        retry_prompt = (
            f"Твой предыдущий ответ нарушает следующие инварианты проекта:\n"
            f"{violation_list}\n\n"
            f"Перепиши ответ, устранив все нарушения. "
            f"Не используй запрещённые инструменты и зависимости.\n\n"
            f"Исходный ответ:\n{answer}"
        )
        payload: dict = {"model": spec.model, "input": retry_prompt}
        if spec.temperature is not None:
            payload["temperature"] = spec.temperature

        try:
            data, _ = self.agent._post(payload)
            if isinstance(data, dict) and not data.get("error"):
                retry_text = _extract_text(data)
                passed, retry_violations = self.agent.ltm.checker.check(retry_text)
                if passed:
                    return retry_text
                vlist = "\n".join(f"  - {v}" for v in retry_violations)
                return (
                    f"Не удалось сформировать ответ без нарушений инвариантов.\n"
                    f"Нарушения:\n{vlist}"
                )
        except Exception:
            pass

        vlist = "\n".join(f"  - {v}" for v in violations)
        return (
            f"Ответ нарушает инварианты проекта и не может быть показан.\n"
            f"Нарушения:\n{vlist}"
        )

    def _agent_names(self) -> str:
        """Comma-separated list of all known agent names."""
        return ", ".join(s.name for s in self.registry.list_all())


# ---------------- Module-level helpers ----------------

def _extract_text(data: dict) -> str:
    """Extract reply text from an OpenAI Responses API response dict."""
    try:
        output = data.get("output", [])
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                content = item.get("content", [])
                if content and isinstance(content, list):
                    return content[0].get("text", "")
        # Fallback: try legacy shape
        return data["output"][0]["content"][0]["text"]
    except Exception:
        return json.dumps(data, ensure_ascii=False, indent=2)


def _planning_violation_message(violations: list[str]) -> str:
    """Return a correction message for PLANNING state invariant violations."""
    lines = ["Ответ содержит нарушения инвариантов проекта:\n"]
    for i, v in enumerate(violations, 1):
        lines.append(f"  Правило {i}: «{v}»")
    lines.append(
        "\nПожалуйста, переформулируй план без использования "
        "запрещённых инструментов и зависимостей.\n"
        "Ограничения: только requests, venv, без SDK и внешних фреймворков."
    )
    return "\n".join(lines)


def _update_facts(facts: dict, user_text: str):
    """Parse 'Key: Value' lines from user message and update facts store."""
    for line in user_text.splitlines():
        if ": " in line:
            key, _, value = line.partition(": ")
            key = key.strip()
            value = value.strip()
            if key:
                facts[key] = value


def _compute_cost(pricing: dict, model: str, in_tok, out_tok):
    """Compute cost from token usage and pricing table."""
    from agent import compute_cost
    return compute_cost(pricing, model, in_tok, out_tok)


def _zero_metrics(model: str, provider: str = "openai") -> dict:
    """Return empty metrics dict."""
    return {
        "provider": provider,
        "model": model,
        "agent": "none",
        "time": 0.0,
        "in": 0,
        "out": 0,
        "cost": "$0.000000",
        "pre_violations": [],
    }


def _fallback_spec(model: str, system_prompt: str) -> AgentSpec:
    """Return a minimal AgentSpec when no agent matches the current state."""
    from agents import AgentSpec, ContextPolicy
    return AgentSpec(
        name="fallback",
        mode="primary",
        description="Fallback agent — no spec matched current state",
        model=model,
        temperature=None,
        when_to_use="fallback only",
        allowed_states=["PLANNING", "EXECUTION", "VALIDATION", "DONE"],
        context_policy=ContextPolicy(
            include_profile=True,
            include_invariants=True,
            include_project_memory=True,
            include_state=True,
            include_history=True,
            history_limit=6,
            include_rules_block=True,
            include_validation_block=False,
            include_summary=True,
        ),
        prompt=system_prompt,
    )
