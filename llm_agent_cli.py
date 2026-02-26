#!/usr/bin/env python3
"""
Simple LLM Agent (CLI)

System prompt file (default):
  ./system_prompt.txt

Example workflow:

1) Start agent:
   $ python llm_agent_cli.py

2) Set goal:
   > /goal Write a learning plan for LLM basics

3) Switch to planning stage:
   > /stage PLAN
   > Create a 5-step plan to reach the goal

4) Execute first step:
   > /stage EXECUTE
   > Start with step 1

5) Review result:
   > /stage REVIEW
   > Check if the plan is realistic

6) Save state:
   > /save

7) Exit:
   > /exit
"""

import os
import sys
import argparse
from pathlib import Path

from agent import Agent, load_pricing_models


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

    p.add_argument("--history-limit", type=int, default=12)
    p.add_argument("--disable-summary", action="store_true")
    p.add_argument("--state", default="state.toon")

    p.add_argument("--system-file", default="system_prompt.txt")
    p.add_argument("--system", help="Override system prompt")

    return p


def print_help():
    """Print compact CLI help with example workflow."""
    print(
        """
Commands:
  /exit    Exit the agent
  /help    Show this help
  /reset   Clear all state and history
  /save    Save state to state.toon
  /load    Load state from state.toon
  /goal    Set high-level agent goal
  /stage   Set stage: IDLE PLAN EXECUTE REVIEW
  /system  Override system prompt temporarily
  /show    Display current stage and goal

System prompt file:
  ./system_prompt.txt

Example workflow:
  /goal Learn LLM fundamentals
  /stage PLAN
  Create a 5-step study plan
  /stage EXECUTE
  Start with step 1
  /stage REVIEW
  Review the result
  /save
"""
    )


def print_metrics(m: dict):
    """Print metrics."""
    print(f"\n@@@MODEL@@@ {m['model']}")
    print(f"@@@M@@@ t={m['time']:.2f}s in={m['in']} out={m['out']} $={m['cost']}")


def main():
    """Run REPL."""
    args = build_parser().parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set")

    pricing = load_pricing_models()

    base_prompt = load_system_prompt(args.system_file)
    system_prompt = args.system if args.system else base_prompt

    agent = Agent(
        api_key=api_key,
        model=args.model,
        system_prompt=system_prompt,
        history_limit=args.history_limit,
        timeout=args.timeout,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        stop=args.stop,
        pricing=pricing,
        print_json=args.json,
        enable_summary=(not args.disable_summary),
    )

    # Auto-load state on startup (if file exists)
    try:
        if Path(args.state).exists():
            agent.load_state(args.state)
            print(f"OK: auto-loaded state from {args.state}")
    except Exception as e:
        print(f"WARNING: could not auto-load state: {e}", file=sys.stderr)

    print("LLM Agent (TOON v3.0).\nType /help\n")

    while True:
        try:
            text = input("> ").strip()
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
                agent.reset()
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
            if text.startswith("/stage "):
                agent.set_stage(text[7:].strip())
                print(f"OK: stage={agent.stage}")
                continue
            if text.startswith("/system "):
                agent.set_system_prompt(text[8:].strip())
                print("OK: system overridden")
                continue
            if text == "/show":
                print("stage:", agent.stage)
                print("goal:", agent.goal)
                print("history:", len(agent.history))
                continue
            print("Unknown command")
            continue

        try:
            answer, metrics = agent.reply(text)

            # Auto-save state after each successful turn
            try:
                agent.save_state(args.state)
            except Exception as e:
                print(f"WARNING: could not auto-save state: {e}", file=sys.stderr)

            print("\n" + answer + "\n")
            print_metrics(metrics)

        except Exception as e:
            print("ERROR:", e, file=sys.stderr)
            try:
                agent.save_state(args.state)
            except Exception:
                pass

    print("Bye.")


if __name__ == "__main__":
    main()
