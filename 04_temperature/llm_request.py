#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
from pathlib import Path

import requests

URL = "https://api.openai.com/v1/responses"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal OpenAI Responses API CLI (no SDK)")
    p.add_argument("prompt", nargs="*", help="Prompt text (if omitted, reads from stdin)")
    p.add_argument("-f", "--prompt-file", help="Read prompt from a text file (UTF-8)")
    p.add_argument("-m", "--model", default="gpt-4.1", help="Model name (default: gpt-4.1)")
    p.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (e.g., 0, 0.7, 1.2)",
    )
    p.add_argument("--max-output-tokens", type=int, default=None, help="Limit output tokens")
    p.add_argument(
        "--stop",
        action="append",
        default=None,
        help="Stop sequence (can be repeated: --stop 'END' --stop '###')",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds (default: 60)",
    )
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

    # If no prompt args, try stdin (supports pipes)
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text

    return "Say hello"


def load_pricing_models() -> dict:
    """
    Loads pricing from pricing.json placed next to this script:
    {
      "models": { "gpt-4.1": {"input": 2.0, "output": 8.0}, ... }
    }
    Returns dict: {model_name: {"input": float, "output": float}, ...}
    """
    path = Path(__file__).resolve().parent / "pricing.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        models = data.get("models", {})
        return models if isinstance(models, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def format_money(x: float) -> str:
    return f"${x:.6f}"


def compute_cost(pricing_models: dict, model: str, input_tokens: int | None, output_tokens: int | None):
    pricing = pricing_models.get(model)

    if pricing is None:
        return "unknown", "unknown", "unknown"

    in_price = pricing.get("input")
    out_price = pricing.get("output")

    if in_price == 0 and out_price == 0:
        return "free", "free", "free"

    if input_tokens is None or output_tokens is None:
        return "unknown", "unknown", "unknown"

    try:
        in_cost = input_tokens * float(in_price) / 1_000_000
        out_cost = output_tokens * float(out_price) / 1_000_000
    except Exception:
        return "unknown", "unknown", "unknown"

    total = in_cost + out_cost
    return format_money(in_cost), format_money(out_cost), format_money(total)


def main() -> None:
    args = build_parser().parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY is not set")

    pricing_models = load_pricing_models()

    prompt = read_prompt(args)

    payload = {
        "model": args.model,
        "input": prompt,
    }

    if args.temperature is not None:
        payload["temperature"] = args.temperature

    if args.max_output_tokens is not None:
        payload["max_output_tokens"] = args.max_output_tokens

    if args.stop:
        payload["stop"] = args.stop

    retries = 3
    delay = 2

    start = time.monotonic()

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=args.timeout,
            )
            break

        except requests.exceptions.ReadTimeout:
            if attempt == retries:
                raise

            print(f"Timeout, retry {attempt}/{retries}...", file=sys.stderr)
            time.sleep(delay)
            delay *= 2

    elapsed = time.monotonic() - start

    data = r.json()

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    if r.status_code != 200:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        raise SystemExit(f"HTTP {r.status_code}")

    # Print model output
    try:
        print(data["output"][0]["content"][0]["text"])
    except Exception:
        print(json.dumps(data, ensure_ascii=False, indent=2))

    # Usage
    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    in_cost, out_cost, total_cost = compute_cost(pricing_models, args.model, input_tokens, output_tokens)

    # Model line (only name)
    print(f"\n@@@MODEL@@@ {args.model}")

    # Compact metrics line
    print(
        f"@@@M@@@ "
        f"t={elapsed:.2f}s "
        f"in={input_tokens} "
        f"out={output_tokens} "
        f"tot={total_tokens} "
        f"$={total_cost}"
    )


if __name__ == "__main__":
    main()
