#!/usr/bin/env python3
import os
import sys
import json
import argparse
import requests

URL = "https://api.openai.com/v1/responses"

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal OpenAI Responses API CLI (no SDK)")
    p.add_argument("prompt", nargs="*", help="Prompt text (if omitted, reads from stdin)")
    p.add_argument("-f", "--prompt-file", help="Read prompt from a text file (UTF-8)")
    p.add_argument("-m", "--model", default="gpt-4.1", help="Model name (default: gpt-4.1)")
    p.add_argument("-t", "--temperature", type=float, default=None, help="Sampling temperature (e.g., 0, 0.7, 1.2)")
    p.add_argument("--max-output-tokens", type=int, default=None, help="Limit output tokens")
    p.add_argument("--stop", action="append", default=None,
                   help="Stop sequence (can be repeated: --stop 'END' --stop '###')")
    p.add_argument("--json", action="store_true", help="Print raw JSON response")
    return p

def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                return text

    if args.prompt:
        return " ".join(args.prompt)

    # If no prompt args, try stdin (supports pipes; heredoc if available)
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text

    return "Say hello"

def main() -> None:
    args = build_parser().parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY is not set")

    prompt = read_prompt(args)

    payload = {
        "model": args.model,
        "input": prompt,
    }
    # Only include optional fields if provided
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.max_output_tokens is not None:
        payload["max_output_tokens"] = args.max_output_tokens
    if args.stop:
        payload["stop"] = args.stop

    r = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    data = r.json()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    if r.status_code != 200:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        raise SystemExit(f"HTTP {r.status_code}")

    # Minimal text extraction (typical case)
    try:
        print(data["output"][0]["content"][0]["text"])
    except Exception:
        # Fallback: print raw JSON if shape differs
        print(json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
