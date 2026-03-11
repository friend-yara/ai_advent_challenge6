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
from orchestrator import Orchestrator

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
  /state <s>               Transition state: PLANNING|PLAN, EXECUTION|EXEC,
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

Prompt format: [STATE:agent] >
  Example: [PLAN:planner] > or [EXEC:coder] >

Agent auto-selection:
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


def _start_reminder_poller(job_id: str, delay_seconds: int, mcp: MCPClient):
    """
    Start a daemon thread that polls reminder status and notifies the user
    when it fires. Uses patch_stdout so prompt_toolkit renders it cleanly.
    """
    server_name = mcp.find_tool_server("reminder")
    if server_name is None:
        return  # reminder tool not available in MCP cache

    def _poll():
        deadline = time.monotonic() + delay_seconds + 30  # grace period
        while time.monotonic() < deadline:
            time.sleep(2)
            try:
                result_dict = mcp.call_tool(server_name, "reminder", {"job_id": job_id})
                status = result_dict.get("data", {}).get("status")
                if status == "completed":
                    content = result_dict.get("content", [])
                    msg_text = content[0].get("text", "") if content else str(result_dict)
                    msg = f"\n🔔 {msg_text}\n"
                    if patch_stdout is not None:
                        with patch_stdout():
                            print(msg, flush=True)
                    else:
                        print(msg, flush=True)
                    return
            except Exception:
                pass  # transient errors — keep polling

    t = threading.Thread(target=_poll, daemon=True,
                         name=f"reminder-poller-{job_id}")
    t.start()


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

    print(f"Agents loaded: {registry.summary()}")
    print(f"MCP servers:   {mcp.summary()}")
    print(f"MCP tools:     {mcp.tools_summary()}")

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
    if state_was_loaded and (agent.tc.task or agent.tc.state != "PLANNING"):
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

    while True:
        try:
            prompt_str = state_prompt(agent.tc, orchestrator)
            if session is not None:
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
                        print("/state PLAN — начать новую задачу")
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
            print("Unknown command")
            continue

        try:
            answer, metrics = orchestrator.reply(text)

            # Print pre-check warnings if any banned patterns were detected
            pre_violations = metrics.get("pre_violations", [])
            if pre_violations:
                print("[ВНИМАНИЕ] Запрос содержит упоминание "
                      "запрещённых правил:")
                for v in pre_violations:
                    print(f"  - {v}")

            # In PLANNING: detect if LLM produced a todo checklist
            if agent.tc.state == "PLANNING":
                steps = agent.plan_from_reply(answer)
                if steps:
                    agent.tc.plan = steps
                    agent.tc.total = len(steps)
                    agent.tc.step = 0
                    agent.tc.done = []
                    agent.tc.current = steps[0]
                    print(answer, end="")
                    print_metrics(metrics)
                    print("Готов перейти к выполнению. "
                          "Введите /state EXEC или внесите корректировки.")
                else:
                    print(answer, end="")
                    print_metrics(metrics)
            else:
                print(answer, end="")
                print_metrics(metrics)

            # Start reminder poller if a reminder tool was called
            for tr in metrics.get("tool_results", []):
                if tr.get("tool_name") == "reminder":
                    # Extract job_id from result text (pattern: "job_id: <hex>")
                    import re as _re
                    m = _re.search(r"job_id[:\s]+([0-9a-f]{12})", tr.get("result_text", ""))
                    if m:
                        _jid = m.group(1)
                        _delay = tr.get("arguments", {}).get("delay_seconds", 30)
                        _start_reminder_poller(_jid, int(_delay), mcp)

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
