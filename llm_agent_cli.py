#!/usr/bin/env python3
"""
LLM Agent CLI — orchestrator-based REPL.

Demo flow (orchestrator + agents):
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

  /agent research    — pin research agent for next message only
  /agent list        — list all available agents
"""

import os
import sys
import argparse
import threading
import time
from pathlib import Path

from agent import Agent, LongTermMemory, load_pricing_models
from agents import AgentRegistry
from context_builder import ContextBuilder
from mcp_client import MCPClient
from orchestrator import Orchestrator, rag_available

# Optional multiline input (prompt_toolkit)
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout
except Exception:
    PromptSession = None
    patch_stdout = None


def load_system_prompt(path: str) -> str:
    """Load system prompt from file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return p.read_text(encoding="utf-8").strip()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    p = argparse.ArgumentParser("Orchestrator-based LLM agent")

    p.add_argument("-m", "--model", default="gpt-4.1")
    p.add_argument("-t", "--temperature", type=float)
    p.add_argument("--max-output-tokens", type=int)
    p.add_argument("--stop", action="append")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--context-strategy",
        default="window",
        choices=["window", "facts", "branch"],
    )
    p.add_argument("--history-limit", type=int, default=6)
    p.add_argument("--context-summary", action="store_true")

    # Profile directory (all per-user files live here)
    p.add_argument("--profile", default="default",
                   help="Profile name (subdirectory under profiles/)")

    # Per-file overrides (default: derived from --profile)
    p.add_argument("--state", default=None)
    p.add_argument("--short-term-file", default=None)
    p.add_argument("--no-auto-load-short-term", action="store_true")
    p.add_argument("--no-auto-save-short-term", action="store_true")

    p.add_argument("--system-file", default="system_prompt.txt")
    p.add_argument("--system", help="Override system prompt")

    # Long-term memory file overrides (default: derived from --profile)
    p.add_argument("--project-memory-file", default=None)
    p.add_argument("--invariants-file", default=None)
    p.add_argument("--use-project-memory", action="store_true", default=True)
    p.add_argument("--use-invariants", action="store_true", default=True)

    # Agents directory
    p.add_argument("--agents-dir", default="agents",
                   help="Directory containing agent spec *.md files")

    # MCP tools directory
    p.add_argument("--tools-dir", default="tools",
                   help="Directory containing MCP server spec *.yaml files")

    p.add_argument("--batch-file", default=None,
                   help="Read REPL input from file instead of terminal")
    p.add_argument("--interactive-batch", default=None,
                   help="Like --batch-file but prompts for follow-up after each answer")

    return p


def resolve_profile_paths(args: argparse.Namespace) -> argparse.Namespace:
    """
    Fill in any file paths not explicitly set by deriving them from
    profiles/<profile>/. Missing files in the profile dir are silently
    tolerated by the loaders that use them.
    """
    profile_dir = Path("profiles") / args.profile
    if args.state is None:
        args.state = str(profile_dir / "state.toon")
    if args.short_term_file is None:
        args.short_term_file = str(profile_dir / "short_term.toon")
    # profile_file is always derived from the profile dir, never overridable
    args.profile_file = str(profile_dir / "PROFILE.json")
    if args.project_memory_file is None:
        args.project_memory_file = str(profile_dir / "PROJECT_MEMORY.md")
    if args.invariants_file is None:
        args.invariants_file = str(profile_dir / "INVARIANTS.yaml")
    return args


def state_prompt(tc, orchestrator: Orchestrator) -> str:
    """Return state-labeled prompt with active agent name, e.g. '[PLAN:planner] > '."""
    labels = {
        "CHAT":       "CHAT",
        "PLANNING":   "PLAN",
        "EXECUTION":  "EXEC",
        "VALIDATION": "VALI",
        "DONE":       "DONE",
    }
    label = labels.get(tc.state, tc.state[:4])
    agent_name = orchestrator.current_agent_name()
    return f"[{label}:{agent_name}] > "


def print_help():
    """Print compact CLI help."""
    print(
        """
Commands:
  /exit                    Exit (progress saved automatically)
  /help                    Show this help
  /reset                   Clear working memory + dialogue, save both files
  /save                    Save working state to state.toon
  /load                    Load working state from state.toon
  /goal <text>             Set task description (alias for /task)
  /task <text>             Set task in working memory
  /state <s>               Transition state: CHAT, PLANNING|PLAN, EXECUTION|EXEC,
                           VALIDATION|VALI, DONE
  /step                    Execute next step (EXECUTION state only)
  /agent <name>            Pin agent for next message only
  /agent list              List all available agents
  /system <text>           Override system prompt temporarily
  /show                    Display working memory + STM status
  /checkpoint              Save snapshot of current branch
  /branch list             List all branches (* = active)
  /branch create <name>    Create new branch from current state
  /branch switch <name>    Switch to branch, saving current first
  /ltm reload              Reload LTM files from disk without restarting
  /whoami                  Short LLM-generated summary of current profile
  /tool list               List all configured MCP servers
  /tool list <server>      Connect to MCP server and show available tools
  /rag                     Show RAG status (mode, threshold, keyword)
  /rag base|filter|off     Switch RAG mode
  /rag threshold <float>   Set similarity threshold (default 0.45)
  /rag keyword on|off      Enable/disable keyword filter (default off)

Prompt format: [STATE:agent] >
  Example: [PLAN:planner] > or [EXEC:coder] >

Agent auto-selection:
  CHAT      → assistant
  PLANNING  → planner
  EXECUTION → coder
  VALIDATION → validator
  (override with /agent <name> for one message)

Invariant checks:
  - pre-check: warns if query matches a banned pattern
  - post-check (PLANNING): replaces violating answers with correction message
  - post-check (other): retries LLM once with correction prompt
  - on /state VALIDATION: auto-checks plan against invariants
"""
    )


def print_metrics(m: dict):
    """Print turn metrics: token counts and cost only."""
    print(f" (in={m['in']}, out={m['out']}, cost={m['cost']})\n")


def _print_cli(msg: str) -> None:
    """Print a message cleanly over the prompt_toolkit prompt if available."""
    if patch_stdout is not None:
        with patch_stdout():
            print(msg, flush=True)
    else:
        print(msg, flush=True)


def _print_rag_answer(answer: str, rag_results: list, rag_metadata: dict) -> None:
    """Print structured RAG output: Answer / Sources / Quotes.

    If rag_results is empty (weak context), prints answer only — no sections.
    """
    from rag.retriever import extract_quote

    print(answer, end="")

    if not rag_results:
        return

    # Sources — deduplicated by (filename, section), ordered by score desc
    print()
    print("Sources:")
    seen_sources: set[tuple] = set()
    ordered: list[dict] = sorted(rag_results, key=lambda r: r.get("score", 0.0), reverse=True)
    for r in ordered:
        key = (r.get("filename", r.get("source", "?")), r.get("section", ""))
        if key in seen_sources:
            continue
        seen_sources.add(key)
        fname = r.get("filename", r.get("source", "?"))
        section = r.get("section") or ""
        chunk_id = r.get("id", "")
        parts = [fname]
        if section:
            parts.append(section)
        if chunk_id:
            parts.append(chunk_id)
        print(f"  - {' | '.join(parts)}")

    # Quotes — one per unique chunk, grouped by filename
    print()
    printed_ids: set[str] = set()
    for r in ordered:
        chunk_id = r.get("id", "")
        if chunk_id in printed_ids:
            continue
        printed_ids.add(chunk_id)
        fname = r.get("filename", r.get("source", "?"))
        section = r.get("section") or ""
        label_parts = [fname]
        if section:
            label_parts.append(section)
        if chunk_id:
            label_parts.append(chunk_id)
        label = " | ".join(label_parts)
        quote = extract_quote(r.get("text", ""))
        print(f'Quotes ({label}):')
        print(f'  "{quote}"')
        print()


def _execute_pipeline(pipeline: list[dict], job_id: str, orch):
    """Delegate pipeline execution to the orchestrator."""
    orch.execute_pipeline(pipeline, job_id, _print_cli)


def _start_reminder_poller(job_id: str, delay_seconds: int, orch):
    """
    Start a daemon thread that polls reminder status and notifies the user
    when it fires. If the reminder has a pipeline, executes it via orchestrator.
    Uses patch_stdout so prompt_toolkit renders it cleanly.
    """
    if orch.mcp is None:
        return  # no MCP configured
    server_name = orch.mcp.find_tool_server("reminder")
    if server_name is None:
        return  # reminder tool not available in MCP cache

    def _poll():
        deadline = time.monotonic() + delay_seconds + 30  # grace period
        while time.monotonic() < deadline:
            time.sleep(2)
            try:
                result_dict = orch.mcp.call_tool(server_name, "reminder", {"job_id": job_id})
                data = result_dict.get("data", {})
                if data.get("status") == "completed":
                    pipeline = data.get("pipeline")
                    if pipeline:
                        _execute_pipeline(pipeline, job_id, orch)
                    else:
                        reminder_text = data.get("text", "")
                        _print_cli(f"\nscheduler:reminder: {reminder_text} - выполнено (job #{job_id})\n")
                    return
            except Exception:
                pass  # transient errors — keep polling

    t = threading.Thread(target=_poll, daemon=True,
                         name=f"reminder-poller-{job_id}")
    t.start()


def _run_indexing(chunker: str) -> None:
    """Build and persist the RAG index for the given chunker strategy."""
    from pathlib import Path as _Path
    try:
        from rag.indexer import build_index
        from rag.retriever import _DEFAULT_INDEX_DIR, _cache
    except ImportError as e:
        print(f"[ERROR] RAG deps missing: {e}. Run: pip install faiss-cpu numpy")
        return

    corpus_path = _Path(__file__).parent / "docs"
    if not corpus_path.exists():
        print(f"[ERROR] docs/ directory not found: {corpus_path}", file=sys.stderr)
        return

    print(f"Индексирую корпус (стратегия: {chunker})...")
    try:
        metrics = build_index(corpus_path, chunker, _DEFAULT_INDEX_DIR)
        # Invalidate in-memory cache so next search reloads from disk
        _cache.clear()
        print(f"  Чанков: {metrics['count']}")
        print(f"  Средний размер: {metrics['avg_chars']:.0f} симв. | "
              f"Мин: {metrics['min_chars']} | Макс: {metrics['max_chars']} | "
              f"σ: {metrics['std_chars']:.0f}")
        print(f"  Источников: {metrics['unique_sources']}")
        print(f"  Время эмбеддинга: {metrics['embed_seconds']:.1f}s")
        print(f"  Индекс сохранён: rag/index/index_{chunker}.faiss")
        print(f"  Активная стратегия: {chunker}")
    except Exception as e:
        print(f"[ERROR] Индексирование не удалось: {e}")


def main():
    """Run REPL."""
    args = resolve_profile_paths(build_parser().parse_args())

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set")

    pricing = load_pricing_models()

    base_prompt = load_system_prompt(args.system_file)
    system_prompt = args.system if args.system else base_prompt

    # Build and load long-term memory
    ltm = LongTermMemory(
        project_memory_file=args.project_memory_file,
        profile_file=args.profile_file,
        invariants_file=args.invariants_file,
        use_project_memory=args.use_project_memory,
        use_profile=True,
        use_invariants=args.use_invariants,
    )
    ltm.load()

    agent = Agent(
        api_key=api_key,
        model=args.model,
        system_prompt=system_prompt,
        history_limit=args.history_limit,
        context_strategy=args.context_strategy,
        timeout=args.timeout,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        stop=args.stop,
        pricing=pricing,
        print_json=args.json,
        context_summary=args.context_summary,
        ltm=ltm,
    )

    # Build agent registry + orchestrator
    registry = AgentRegistry()
    registry.load(args.agents_dir)
    # Build MCP client (loads server specs + pre-fetches tool schemas)
    mcp = MCPClient(args.tools_dir)
    mcp.load()

    orchestrator = Orchestrator(agent, registry, ContextBuilder(), pricing, mcp=mcp)
    orchestrator.rag_mode = "filter" if rag_available() else "off"

    print(f"Agents loaded: {registry.summary()}")
    print(f"MCP servers:   {mcp.summary()}")
    print(f"MCP tools:     {mcp.tools_summary()}")
    _rag_init = f"режим={orchestrator.rag_mode}" if orchestrator.rag_mode != "off" else "выключен (нет индекса)"
    print(f"RAG:           {_rag_init}")

    # Auto-load working state on startup (if file exists)
    state_was_loaded = False
    try:
        if Path(args.state).exists():
            agent.load_state(args.state)
            state_was_loaded = True
    except Exception as e:
        print(f"WARNING: could not auto-load state: {e}", file=sys.stderr)

    # Auto-load short-term memory on startup (if file exists and not disabled)
    if not args.no_auto_load_short_term:
        try:
            if Path(args.short_term_file).exists():
                agent.load_short_term(args.short_term_file)
        except Exception as e:
            print(f"WARNING: could not auto-load short-term: {e}", file=sys.stderr)

    print(f"LLM Agent (TOON v3.0). Model={agent.model}, "
          f"strategy={agent.context_strategy}, "
          f"history-limit={agent.history_limit}")
    profile_name = (
        agent.ltm.profile_obj.data.get("meta", {}).get("id", "unknown")
        if agent.ltm.profile_obj else "unknown"
    )

    # Welcome-back or fresh greeting
    if state_was_loaded and (agent.tc.task or agent.tc.state not in ("PLANNING", "CHAT")):
        try:
            print(agent.welcome_back())
        except Exception:
            print(f"С возвращением, {profile_name}!")
        tc = agent.tc
        if tc.state == "PLANNING" and tc.plan:
            print(agent.format_todo())
            print("Введите /state EXEC чтобы перейти к выполнению, "
                  "или продолжите корректировку плана.")
        elif tc.state == "EXECUTION":
            print(agent.format_todo())
            step_num = tc.step + 1
            print(f"Вы остановились на шаге {step_num}: {tc.current}")
            print("/step — продолжить | /state PLAN — вернуться к планированию"
                  " | /exit — выйти")
        elif tc.state == "VALIDATION":
            print("Вы в стадии VALIDATION. Продолжайте проверку или введите "
                  "/state EXEC / /state DONE.")
        elif tc.state == "DONE":
            print("Предыдущая задача завершена. /state PLAN — новая задача.")
    else:
        print(f"Привет, {profile_name}!")

    print("Подсказка: Enter — новая строка, Esc+Enter — отправить, "
          "Ctrl+D — выход.\n")

    session = None
    if PromptSession is not None:
        session = PromptSession(mouse_support=False)

    _batch_iter = None
    _interactive_batch_iter = None
    _interactive_batch_followup = False  # True when processing a follow-up after a batch question
    if args.batch_file:
        _batch_fh = open(args.batch_file, encoding="utf-8")
        _batch_iter = iter(_batch_fh)
    if getattr(args, "interactive_batch", None):
        _ibatch_fh = open(args.interactive_batch, encoding="utf-8")
        _interactive_batch_iter = iter(_ibatch_fh)

    while True:
        try:
            prompt_str = state_prompt(agent.tc, orchestrator)
            if _batch_iter is not None:
                try:
                    text = next(_batch_iter).strip()
                    print(f"{prompt_str}{text}")
                except StopIteration:
                    break
            elif _interactive_batch_iter is not None and not _interactive_batch_followup:
                try:
                    text = next(_interactive_batch_iter).strip()
                    print(f"{prompt_str}{text}")
                    _interactive_batch_followup = True  # will prompt for follow-up after answer
                except StopIteration:
                    break
            elif _interactive_batch_iter is not None and _interactive_batch_followup:
                # Prompt for optional follow-up after each batch answer
                _fu_prompt = "[follow-up or Enter to continue]: "
                if session is not None:
                    with patch_stdout():
                        text = session.prompt(_fu_prompt, multiline=False).strip()
                else:
                    text = input(_fu_prompt).strip()
                if not text:
                    _interactive_batch_followup = False  # no follow-up → next batch question
                    continue
            elif session is not None:
                with patch_stdout():
                    text = session.prompt(
                        prompt_str,
                        multiline=True,
                        prompt_continuation="... ",
                    ).strip()
            else:
                text = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            continue

        if text.startswith("/"):
            if text == "/exit":
                break
            if text == "/help":
                print_help()
                continue
            if text == "/reset":
                agent.reset_working()
                agent.reset_short_term()
                agent.save_state(args.state)
                agent.save_short_term(args.short_term_file)
                print("OK: reset")
                continue
            if text == "/save":
                agent.save_state(args.state)
                print("OK: saved")
                continue
            if text == "/load":
                agent.load_state(args.state)
                print("OK: loaded")
                continue
            if text.startswith("/goal "):
                agent.set_goal(text[6:].strip())
                print("OK: goal set")
                continue
            if text.startswith("/task "):
                agent.tc.task = text[6:].strip()
                print("OK: task set")
                continue
            if text.startswith("/state "):
                raw = text[7:].strip().upper()
                _aliases = {
                    "EXEC": "EXECUTION",
                    "PLAN": "PLANNING",
                    "VALID": "VALIDATION",
                    "VALI": "VALIDATION",
                }
                new_state = _aliases.get(raw, raw)
                # Guard: plan must exist before entering EXECUTION
                if (new_state == "EXECUTION"
                        and not agent.tc.can_transition_to_execution()):
                    print("ERROR: план пуст. Сначала сформируйте план "
                          "в PLANNING.")
                    continue
                err = agent.set_task_state(new_state)
                if err:
                    print(f"ERROR: {err}")
                else:
                    print(f"OK: state={agent.tc.state}")
                    if agent.tc.state == "EXECUTION":
                        print(agent.format_todo())
                        print(f"Текущий шаг: {agent.tc.current}")
                        print("/step — выполнить шаг | "
                              "/state PLAN — к планированию | "
                              "/exit — выйти")
                    elif agent.tc.state == "VALIDATION":
                        # Auto-check plan steps against invariants + profile
                        check_parts = agent.tc.plan + agent.tc.done
                        check_text = "\n".join(check_parts)

                        all_passed = True

                        if not check_parts:
                            print("[VALIDATION] План пуст — нечего проверять.")
                            all_passed = False
                        else:
                            # Check against INVARIANTS banned rules
                            if agent.ltm.invariants and agent.ltm.checker.rules:
                                v_passed, v_violations = (
                                    agent.ltm.checker.check(check_text)
                                )
                                if not v_passed:
                                    all_passed = False
                                    print("[VALIDATION] Нарушения требований "
                                          "и ограничений:")
                                    for v in v_violations:
                                        print(f"  - {v}")

                            # Check against profile constraints (same checker,
                            # covers sdk/gui/framework bans from profile too)
                            if (agent.ltm.profile_obj
                                    and agent.ltm.profile_obj.enabled
                                    and agent.ltm.checker.rules):
                                # Profile constraints overlap with invariants;
                                # run a separate check only if invariants were
                                # not loaded (avoid duplicate output)
                                if not agent.ltm.invariants:
                                    p_passed, p_violations = (
                                        agent.ltm.checker.check(check_text)
                                    )
                                    if not p_passed:
                                        all_passed = False
                                        print("[VALIDATION] Нарушения "
                                              "ограничений профиля:")
                                        for v in p_violations:
                                            print(f"  - {v}")

                        if all_passed:
                            print(
                                "Проверка завершена, требования и ограничения "
                                "соблюдены. Рекомендую перейти к завершению "
                                "задачи."
                            )
                        else:
                            print("Устраните нарушения перед переходом в DONE.")

                        print("/state EXEC — исправить | /state DONE — завершить")
                    elif agent.tc.state == "DONE":
                        print("Задача завершена. Результаты ИИ следует "
                              "перепроверять вручную.")
                        agent.set_task_state("CHAT")
                        print(f"OK: state={agent.tc.state}")
                        print("Возврат в режим чата. "
                              "/task + /state PLAN — новая задача.")
                    agent.save_state(args.state)
                continue
            if text == "/step":
                if agent.tc.state != "EXECUTION":
                    print("ERROR: /step доступен только в состоянии EXECUTION")
                    continue
                if agent.tc.step >= agent.tc.total:
                    print("Все шаги уже выполнены.")
                    print("/state VALID — валидация | "
                          "/state PLAN — перепланировать")
                    continue
                try:
                    answer, metrics = orchestrator.run_step()
                    print(answer)
                    print()
                    print(agent.format_todo())
                    if agent.tc.step >= agent.tc.total:
                        print("\nВсе шаги выполнены.")
                        print("/state VALID — валидация | "
                              "/state PLAN — перепланировать")
                    else:
                        print(f"\nСледующий шаг: {agent.tc.current}")
                        print("/step — выполнить | "
                              "/state PLAN — к планированию | "
                              "/exit — выйти")
                    print_metrics(metrics)
                    agent.save_state(args.state)
                    if not args.no_auto_save_short_term:
                        agent.save_short_term(args.short_term_file)
                except Exception as e:
                    print(f"ERROR: {e}", file=sys.stderr)
                continue
            if text.startswith("/agent"):
                parts = text.split(maxsplit=1)
                sub = parts[1].strip() if len(parts) > 1 else ""
                if sub == "list" or sub == "":
                    for spec in registry.list_all():
                        marker = "*" if spec.mode == "primary" else " "
                        active = (
                            " ← active"
                            if spec.name == orchestrator.current_agent_name()
                            else ""
                        )
                        print(
                            f"  {marker} {spec.name:<12} "
                            f"[{spec.mode}] {spec.model}  "
                            f"states={spec.allowed_states}{active}"
                        )
                else:
                    err = orchestrator.pin_agent(sub)
                    if err:
                        print(f"ERROR: {err}")
                    else:
                        print(f"OK: next message will use agent='{sub}'")
                continue
            if text.startswith("/system "):
                agent.set_system_prompt(text[8:].strip())
                print("OK: system overridden")
                continue
            if text == "/show":
                tc = agent.tc
                print(
                    f"[working] task={tc.task!r}, state={tc.state}, "
                    f"step={tc.step}/{tc.total}, current={tc.current!r}"
                )
                summary_flag = "yes" if agent.stm.summary else "no"
                print(
                    f"[stm]     messages={len(agent.stm.messages)}, "
                    f"summary={summary_flag}, facts={len(agent.facts)}, "
                    f"branch={agent.current_branch}"
                )
                checker_info = (
                    f", checker={agent.ltm.checker.summary_line()}"
                    if agent.ltm.invariants else ""
                )
                print(f"[ltm]     {agent.ltm.summary_line()}{checker_info}")
                print(f"[orch]    agents={registry.summary()}, "
                      f"active={orchestrator.current_agent_name()}")
                print(f"[mcp]     {mcp.summary()}")
                rag_idx = "index=ok" if rag_available() else "index=missing"
                print(f"[rag]     mode={orchestrator.rag_mode}, {rag_idx}")
                continue
            if text == "/checkpoint":
                agent.checkpoint()
                print(f"OK: checkpoint saved to branch '{agent.current_branch}'")
                continue
            if text == "/whoami":
                try:
                    print(agent.whoami(args.profile))
                except Exception as e:
                    print(f"ERROR: {e}", file=sys.stderr)
                continue
            if text == "/ltm reload":
                agent.ltm.reload()
                print(f"OK: LTM reloaded — {agent.ltm.summary_line()}")
                continue
            if text.startswith("/branch"):
                parts = text.split()
                sub = parts[1] if len(parts) > 1 else ""
                if sub == "list":
                    for name in agent.branches:
                        marker = "*" if name == agent.current_branch else " "
                        print(f"  {marker} {name}")
                elif sub == "create" and len(parts) == 3:
                    try:
                        agent.branch_create(parts[2])
                        print(f"OK: branch '{parts[2]}' created")
                    except ValueError as e:
                        print(f"ERROR: {e}")
                elif sub == "switch" and len(parts) == 3:
                    try:
                        agent.branch_switch(parts[2])
                        print(f"OK: switched to branch '{agent.current_branch}'")
                    except ValueError as e:
                        print(f"ERROR: {e}")
                else:
                    print("Usage: /branch list | /branch create <name> | "
                          "/branch switch <name>")
                continue
            if text.startswith("/tool"):
                parts = text.split(maxsplit=2)
                sub = parts[1].strip() if len(parts) > 1 else ""
                srv = parts[2].strip() if len(parts) > 2 else ""
                if sub == "list" and not srv:
                    # List configured servers
                    servers = mcp.list_servers()
                    if not servers:
                        print("Нет настроенных MCP-серверов "
                              f"(каталог: {args.tools_dir})")
                    else:
                        print(f"MCP-серверы ({len(servers)}):")
                        for s in servers:
                            print(f"  {s['name']:<16} {s['url']}")
                            if s.get("description"):
                                print(f"                   {s['description']}")
                elif sub == "list" and srv:
                    # Connect to server and show tools
                    spec = mcp.get_server(srv)
                    if spec is None:
                        known = ", ".join(
                            s["name"] for s in mcp.list_servers()
                        ) or "(нет)"
                        print(f"ERROR: сервер '{srv}' не найден. "
                              f"Доступны: {known}")
                    else:
                        print(f"Сервер: {spec['name']}  ({spec['url']})")
                        if spec.get("description"):
                            print(f"  {spec['description']}")
                        print("Получение списка инструментов...")
                        try:
                            tools = mcp.list_tools(srv)
                            if not tools:
                                print("  (инструментов не найдено)")
                            else:
                                print(f"\nИнструменты ({len(tools)}):\n")
                                for t in tools:
                                    name = t.get("name", "?")
                                    title = t.get("title") or "—"
                                    desc = t.get("description", "")
                                    print(f"  {name}")
                                    print(f"    Title:       {title}")
                                    if desc:
                                        # Wrap long descriptions at 72 chars
                                        words = desc.split()
                                        line, out = "", []
                                        for w in words:
                                            if len(line) + len(w) + 1 > 72:
                                                out.append(line)
                                                line = w
                                            else:
                                                line = (
                                                    f"{line} {w}"
                                                    if line else w
                                                )
                                        if line:
                                            out.append(line)
                                        indent = "    Description: "
                                        cont   = "                 "
                                        for i, ln in enumerate(out):
                                            pfx = indent if i == 0 else cont
                                            print(f"{pfx}{ln}")
                                    print()
                        except Exception as e:
                            print(f"ERROR: {e}", file=sys.stderr)
                else:
                    print("Использование:")
                    print("  /tool list              — список MCP-серверов")
                    print("  /tool list <сервер>     — инструменты сервера")
                continue
            if text.startswith("/index"):
                parts = text.split()
                _chunker = parts[1] if len(parts) > 1 else "fixed"
                if _chunker not in ("fixed", "structured"):
                    print("Использование: /index [fixed|structured]")
                else:
                    _run_indexing(_chunker)
                continue
            if text.startswith("/rag"):
                parts = text.split(maxsplit=2)
                sub = parts[1].strip().lower() if len(parts) > 1 else ""
                if sub in ("base", "filter"):
                    if not rag_available():
                        print("ERROR: RAG-индекс не найден. Запустите /index.")
                    else:
                        orchestrator.rag_mode = sub
                        print(f"OK: RAG режим = {sub}")
                elif sub == "off":
                    orchestrator.rag_mode = "off"
                    print("OK: RAG выключен")
                elif sub == "threshold":
                    val = parts[2].strip() if len(parts) > 2 else ""
                    try:
                        orchestrator.rag_similarity_threshold = float(val)
                        print(f"OK: RAG threshold = {orchestrator.rag_similarity_threshold}")
                    except ValueError:
                        print("Использование: /rag threshold <float>  (например: /rag threshold 0.45)")
                elif sub == "keyword":
                    val = parts[2].strip().lower() if len(parts) > 2 else ""
                    if val == "on":
                        orchestrator.rag_use_keyword_filter = True
                        print("OK: RAG keyword-фильтр включён")
                    elif val == "off":
                        orchestrator.rag_use_keyword_filter = False
                        print("OK: RAG keyword-фильтр выключен")
                    else:
                        print("Использование: /rag keyword on|off")
                elif sub == "min":
                    val = parts[2].strip() if len(parts) > 2 else ""
                    try:
                        orchestrator.rag_min_results = int(val)
                        print(f"OK: RAG min_results = {orchestrator.rag_min_results}")
                    except ValueError:
                        print("Использование: /rag min <int>  (например: /rag min 1)")
                elif sub == "":
                    idx_str = "индекс найден" if rag_available() else "индекс отсутствует"
                    kw = "on" if orchestrator.rag_use_keyword_filter else "off"
                    print(
                        f"RAG: режим={orchestrator.rag_mode}, "
                        f"threshold={orchestrator.rag_similarity_threshold}, "
                        f"keyword={kw}, min_results={orchestrator.rag_min_results} "
                        f"({idx_str})"
                    )
                else:
                    print("Использование: /rag | /rag base|filter|off | /rag threshold <float> | /rag keyword on|off | /rag min <int>")
                continue
            print("Unknown command")
            continue

        try:
            orchestrator.last_rag_results = []
            orchestrator.last_rag_metadata = {}
            answer, metrics = orchestrator.reply(text)

            # Print pre-check warnings if any banned patterns were detected
            pre_violations = metrics.get("pre_violations", [])
            if pre_violations:
                print("[ВНИМАНИЕ] Запрос содержит упоминание "
                      "запрещённых правил:")
                for v in pre_violations:
                    print(f"  - {v}")

            # Detect if RAG document_search was invoked this turn
            _has_rag_results = bool(orchestrator.last_rag_results)

            # In PLANNING: detect if LLM produced a todo checklist
            # In CHAT: never parse plan — print answer directly
            if agent.tc.state == "PLANNING":
                steps = agent.plan_from_reply(answer)
                if steps:
                    agent.tc.plan = steps
                    agent.tc.total = len(steps)
                    agent.tc.step = 0
                    agent.tc.done = []
                    agent.tc.current = steps[0]
                    if _has_rag_results:
                        _print_rag_answer(answer, orchestrator.last_rag_results, orchestrator.last_rag_metadata)
                    else:
                        print(answer, end="")
                    print_metrics(metrics)
                    print("Готов перейти к выполнению. "
                          "Введите /state EXEC или внесите корректировки.")
                else:
                    if _has_rag_results:
                        _print_rag_answer(answer, orchestrator.last_rag_results, orchestrator.last_rag_metadata)
                    else:
                        print(answer, end="")
                    print_metrics(metrics)
            else:
                if _has_rag_results:
                    _print_rag_answer(answer, orchestrator.last_rag_results, orchestrator.last_rag_metadata)
                else:
                    print(answer, end="")
                print_metrics(metrics)

            # Start reminder poller if a reminder tool was called
            for tr in metrics.get("tool_results", []):
                _tname = tr.get("tool_name", "")
                _server = tr.get("server", "")
                _qualified = f"{_server}:{_tname}" if _server else _tname
                if _tname == "reminder":
                    _jid = tr.get("data", {}).get("job_id")
                    if _jid:
                        _reminder_text = tr.get("data", {}).get("text", "")
                        _print_cli(f"scheduler:reminder: запланирована задача «{_reminder_text}» (job #{_jid})")
                        args_d = tr.get("arguments", {})
                        if args_d.get("delay_days"):
                            _delay = int(float(args_d["delay_days"]) * 86400)
                        elif args_d.get("delay_hours"):
                            _delay = int(float(args_d["delay_hours"]) * 3600)
                        elif args_d.get("delay_minutes"):
                            _delay = int(float(args_d["delay_minutes"]) * 60)
                        else:
                            _delay = int(args_d.get("delay_seconds", 30))
                        _start_reminder_poller(_jid, _delay, orchestrator)
                elif _tname == "get_forecast":
                    _print_cli(f"weather:get_forecast: {tr.get('result_text', '').strip()[:120]}")
                elif _tname == "save_to_file":
                    _print_cli(f"storage:save_to_file: {tr.get('result_text', '').strip()[:120]}")
                elif _tname == "document_search":
                    _data = tr.get("data", {})
                    _results = _data.get("results", [])
                    _rag_mode = _data.get("mode", "base")

                    # дедупликация источников: максимальный score на файл
                    _seen: dict[str, float] = {}
                    for _r in _results:
                        _src = _r.get("source", "?")
                        _sc = _r.get("score", 0.0)
                        if _src not in _seen or _sc > _seen[_src]:
                            _seen[_src] = _sc
                    _sources_str = " ".join(
                        f"{s}({sc:.2f})" for s, sc in sorted(_seen.items(), key=lambda x: -x[1])
                    )

                    if _rag_mode == "filter":
                        _orig = _data.get("original_query") or tr.get("arguments", {}).get("query", "")
                        _rewritten = _data.get("rewritten_query")
                        _initial = _data.get("initial_count", 0)
                        _kept = len(_results)
                        _dropped_count = len(_data.get("filtered_out", []))

                        _query_part = f'"{_orig}"'
                        if _rewritten and _rewritten != _orig:
                            _query_part += f' → "{_rewritten}"'

                        _pipeline_part = f"{_initial}→{_kept}"
                        if _dropped_count:
                            _pipeline_part += f" (−{_dropped_count})"

                        _print_cli(f"[RAG:filter] {_query_part} | {_pipeline_part} | {_sources_str}")
                    else:
                        _print_cli(f"[RAG:base] {len(_results)} chunks | {_sources_str}")

            # Auto-save working state after each successful turn
            try:
                agent.save_state(args.state)
            except Exception as e:
                print(f"WARNING: could not auto-save state: {e}",
                      file=sys.stderr)

            # Auto-save short-term memory after each successful turn
            if not args.no_auto_save_short_term:
                try:
                    agent.save_short_term(args.short_term_file)
                except Exception as e:
                    print(f"WARNING: could not auto-save short-term: {e}",
                          file=sys.stderr)

        except Exception as e:
            print("ERROR:", e, file=sys.stderr)
            try:
                agent.save_state(args.state)
            except Exception:
                pass

    print("Bye.")


if __name__ == "__main__":
    main()
