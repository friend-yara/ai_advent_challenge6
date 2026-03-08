#!/usr/bin/env python3
"""
Simple LLM Agent (CLI)

System prompt file (default):
  ./system_prompt.txt

Example workflow:

1) Start agent:
   $ python llm_agent_cli.py

2) Set task:
   > /task Write a learning plan for LLM basics

3) Switch to execution state:
   > /state EXECUTION
   > Start with step 1

4) Validate result:
   > /state VALIDATION
   > Check if the plan is realistic

5) Save state:
   > /save

6) Exit:
   > /exit
"""

import os
import sys
import argparse
from pathlib import Path

from agent import Agent, LongTermMemory, load_pricing_models

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
    p = argparse.ArgumentParser("Stage-based LLM agent")

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


def state_prompt(tc) -> str:
    """Return state-labeled prompt prefix, e.g. '[PLAN] > '."""
    labels = {
        "PLANNING":   "[PLAN]",
        "EXECUTION":  "[EXEC]",
        "VALIDATION": "[VALI]",
        "DONE":       "[DONE]",
    }
    label = labels.get(tc.state, f"[{tc.state[:4]}]")
    return f"{label} > "


def print_help():
    """Print compact CLI help with example workflow."""
    print(
        """
Commands:
  /exit                    Exit the agent (progress saved automatically)
  /help                    Show this help
  /reset                   Clear working memory + dialogue, save both files
  /save                    Save working state to state.toon
  /load                    Load working state from state.toon
  /goal                    Set task description (alias for /task)
  /task <text>             Set task in working memory
  /state <s>               Transition state: PLANNING|EXEC|EXECUTION|
                           VALIDATION|VALI|DONE
  /step                    Execute next step (EXECUTION state only)
  /system                  Override system prompt temporarily
  /show                    Display working memory + STM (two lines)
  /checkpoint              Save snapshot of current branch
  /branch list             List all branches (* = active)
  /branch create <name>    Create new branch from current state
  /branch switch <name>    Switch to branch, saving current first
  /ltm reload              Reload LTM files from disk without restarting
  /whoami                  Short summary of current user profile (<=80 words)

Notes:
  Invariant checks run automatically:
  - pre-check: warns if your query matches a banned pattern
  - post-check (PLANNING): replaces violating LLM answers with a
    correction message
  - post-check (other states): retries LLM once with correction prompt
  - on entering VALIDATION: auto-checks the plan against invariants

Flags:
  --profile <name>           Profile to use (default: default).
                             All files default to profiles/<name>/.
                             Missing files are silently skipped.
  --short-term-file          Override short-term memory file path
  --no-auto-load-short-term  Skip loading short_term.toon at startup
  --no-auto-save-short-term  Skip saving short_term.toon after each turn
  --project-memory-file      Override project memory file path
  --invariants-file          Override invariants file path
  --use-project-memory       Inject PROJECT_MEMORY into prompt
  --use-invariants           Inject INVARIANTS into prompt

System prompt file:
  ./system_prompt.txt
"""
    )

def print_metrics(m: dict):
    """Print metrics."""
    print(f" ({m['time']:.2f} s, in={m['in']}, out={m['out']}, cost={m['cost']})\n")

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
            prompt_str = state_prompt(agent.tc)
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
                    answer = agent.run_step()
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
                    agent.save_state(args.state)
                    if not args.no_auto_save_short_term:
                        agent.save_short_term(args.short_term_file)
                except Exception as e:
                    print(f"ERROR: {e}", file=sys.stderr)
                continue
            if text.startswith("/system "):
                agent.set_system_prompt(text[8:].strip())
                print("OK: system overridden")
                continue
            if text == "/show":
                tc = agent.tc
                print(f"[working] task={tc.task!r}, state={tc.state}, step={tc.step}/{tc.total}, current={tc.current!r}")
                summary_flag = "yes" if agent.stm.summary else "no"
                print(f"[stm]     messages={len(agent.stm.messages)}, summary={summary_flag}, facts={len(agent.facts)}, branch={agent.current_branch}")
                checker_info = (
                    f", checker={agent.ltm.checker.summary_line()}"
                    if agent.ltm.invariants else ""
                )
                print(f"[ltm]     {agent.ltm.summary_line()}{checker_info}")
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
                    print("Usage: /branch list | /branch create <name> | /branch switch <name>")
                continue
            print("Unknown command")
            continue

        try:
            answer, metrics = agent.reply(text)

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
