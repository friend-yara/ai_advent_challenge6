#!/usr/bin/env python3
"""
ai_review.py — Automated AI code review.

Standalone script that analyses a code diff using RAG context (docs + code
review checklist) and an LLM, then outputs a structured review.

Usage:
    python ai_review.py --diff-file pr.diff
    git diff HEAD~1 | python ai_review.py
    python ai_review.py --pr-number 42 --repo user/repo --post-comment

Environment variables:
    OPENAI_API_KEY  — required for LLM calls
    GITHUB_TOKEN    — required for --pr-number and --post-comment
"""

import argparse
import os
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# RAG availability (optional — graceful fallback)
# ---------------------------------------------------------------------------

try:
    from rag.retriever import search_improved
    _RAG_AVAILABLE = True
except Exception:
    _RAG_AVAILABLE = False

# Reuse existing provider
from providers import OpenAIProvider

_SCRIPT_DIR = Path(__file__).resolve().parent
_CHECKLIST_PATH = _SCRIPT_DIR / "docs" / "development-practises" / "code-review.md"
_MAX_DIFF_CHARS = 12_000


# ---------------- Argument parsing ----------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="AI Code Review")
    p.add_argument("--diff-file", type=str, help="Path to diff file")
    p.add_argument("--pr-number", type=int, help="GitHub PR number (fetches diff via API)")
    p.add_argument("--repo", type=str, help="GitHub repo (owner/name) for API calls")
    p.add_argument("--post-comment", action="store_true", help="Post review as PR comment")
    p.add_argument("--model", type=str, default="gpt-4.1-mini", help="LLM model (default: gpt-4.1-mini)")
    p.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens")
    return p.parse_args()


# ---------------- Diff acquisition ----------------

def get_diff(args: argparse.Namespace) -> str:
    """Get diff text from file, GitHub API, or stdin."""
    # 1. From file
    if args.diff_file:
        path = Path(args.diff_file)
        if not path.exists():
            print(f"[ERROR] Diff file not found: {args.diff_file}", file=sys.stderr)
            sys.exit(1)
        return path.read_text(encoding="utf-8")

    # 2. From GitHub API
    if args.pr_number and args.repo:
        return _fetch_pr_diff(args.repo, args.pr_number)

    # 3. From stdin
    if not sys.stdin.isatty():
        return sys.stdin.read()

    print("[ERROR] No diff source. Use --diff-file, --pr-number, or pipe diff to stdin.", file=sys.stderr)
    sys.exit(1)


def _fetch_pr_diff(repo: str, pr_number: int) -> str:
    """Fetch PR diff from GitHub API."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("[ERROR] GITHUB_TOKEN required for --pr-number", file=sys.stderr)
        sys.exit(1)

    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff",
    }
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        print(f"[ERROR] GitHub API {r.status_code}: {r.text[:200]}", file=sys.stderr)
        sys.exit(1)
    return r.text


# ---------------- Diff truncation ----------------

def truncate_diff(diff: str, max_chars: int = _MAX_DIFF_CHARS) -> str:
    """Truncate diff to max_chars, prioritising .py files."""
    if len(diff) <= max_chars:
        return diff

    # Split into per-file sections
    sections: list[tuple[str, str]] = []  # (filename, section_text)
    current_file = ""
    current_lines: list[str] = []

    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git"):
            if current_lines:
                sections.append((current_file, "".join(current_lines)))
            # Extract filename: "diff --git a/foo.py b/foo.py"
            parts = line.split()
            current_file = parts[-1].lstrip("b/") if len(parts) >= 4 else ""
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_file, "".join(current_lines)))

    # Sort: .py files first, then by original order
    py_sections = [(f, t) for f, t in sections if f.endswith(".py")]
    other_sections = [(f, t) for f, t in sections if not f.endswith(".py")]

    result_parts: list[str] = []
    total = 0
    for _fname, text in py_sections + other_sections:
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                result_parts.append(text[:remaining])
            break
        result_parts.append(text)
        total += len(text)

    total_lines = diff.count("\n")
    result = "".join(result_parts)
    result += f"\n\n... (truncated, {total_lines} total lines in diff)"
    return result


# ---------------- Review context (RAG + fallback) ----------------

def get_review_context() -> str:
    """Get review checklist context from RAG or raw file."""
    # Try RAG first
    if _RAG_AVAILABLE:
        try:
            results, _dropped = search_improved(
                "code review checklist bugs security architecture testing",
                initial_top_k=10,
                final_top_k=5,
                threshold=0.25,
            )
            if results:
                print(f"[RAG] Using {len(results)} chunks from docs", file=sys.stderr)
                parts = []
                for r in results:
                    section = f" [{r['section']}]" if r.get("section") else ""
                    parts.append(f"[{r['filename']}]{section}\n{r['text']}")
                return "\n\n---\n\n".join(parts)
        except Exception as e:
            print(f"[RAG] Search failed: {e}", file=sys.stderr)

    # Fallback: read raw checklist file
    if _CHECKLIST_PATH.exists():
        print("[RAG] Fallback to raw checklist", file=sys.stderr)
        return _CHECKLIST_PATH.read_text(encoding="utf-8")

    print("[RAG] No context available", file=sys.stderr)
    return ""


# ---------------- Prompt construction ----------------

_SYSTEM_PROMPT = """\
You are an AI code reviewer. Analyse the provided code diff using the review \
checklist and documentation context. Be specific — reference file names and \
line numbers from the diff.

Structure your review as:

## Potential Bugs
List any bugs, logic errors, off-by-one mistakes, unhandled edge cases.

## Architectural Issues
Note design problems, coupling, missing abstractions, or violations of \
project conventions.

## Recommendations
Suggest improvements: readability, performance, security, testing.

## Summary
One-paragraph overall assessment. If the code looks good, say so explicitly.

Keep the review concise and actionable. Do not repeat the diff back.\
"""


def build_prompt(diff: str, context: str) -> list[dict]:
    """Build LLM message list from diff and review context."""
    user_parts = []
    if context:
        user_parts.append(f"## Review Checklist & Documentation\n\n{context}")
    user_parts.append(f"## Code Diff\n\n```diff\n{diff}\n```")
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


# ---------------- LLM call ----------------

def call_llm(messages: list[dict], model: str, max_tokens: int) -> str:
    """Call OpenAI Responses API and return review text."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    provider = OpenAIProvider(api_key=api_key)
    payload = {
        "model": model,
        "input": messages,
        "temperature": 0.2,
        "max_output_tokens": max_tokens,
    }

    print(f"[LLM] Calling {model}...", file=sys.stderr)
    data, elapsed = provider.post(payload, timeout=120)

    if data.get("error"):
        print(f"[ERROR] API error: {data['error']}", file=sys.stderr)
        sys.exit(1)

    # Extract text from Responses API format
    try:
        text = data["output"][0]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        print(f"[ERROR] Unexpected API response: {data}", file=sys.stderr)
        sys.exit(1)

    usage = data.get("usage", {})
    in_tok = usage.get("input_tokens", "?")
    out_tok = usage.get("output_tokens", "?")
    print(f"[LLM] Done in {elapsed:.1f}s ({in_tok} in / {out_tok} out)", file=sys.stderr)
    return text


# ---------------- GitHub comment ----------------

def post_github_comment(repo: str, pr_number: int, body: str) -> None:
    """Post review as a PR comment via GitHub API."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("[ERROR] GITHUB_TOKEN required for --post-comment", file=sys.stderr)
        return

    comment_body = (
        "<details>\n"
        f"<summary>🤖 AI Code Review (automated)</summary>\n\n"
        f"{body}\n"
        "</details>"
    )

    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    r = requests.post(url, headers=headers, json={"body": comment_body}, timeout=30)
    if r.status_code == 201:
        print(f"[GitHub] Comment posted: {r.json().get('html_url', '')}", file=sys.stderr)
    else:
        print(f"[ERROR] GitHub comment failed {r.status_code}: {r.text[:200]}", file=sys.stderr)


# ---------------- Main ----------------

def main() -> None:
    """Entry point."""
    args = parse_args()

    # 1. Get diff
    diff = get_diff(args)
    if not diff.strip():
        print("[INFO] Empty diff — nothing to review.", file=sys.stderr)
        sys.exit(0)

    # 2. Truncate diff
    diff = truncate_diff(diff)

    # 3. Get review context (RAG or fallback)
    context = get_review_context()

    # 4. Build prompt and call LLM
    messages = build_prompt(diff, context)
    review = call_llm(messages, args.model, args.max_tokens)

    # 5. Output review
    print(review)

    # 6. Post as GitHub comment if requested
    if args.post_comment:
        if not args.pr_number or not args.repo:
            print("[WARN] --post-comment requires --pr-number and --repo", file=sys.stderr)
        else:
            post_github_comment(args.repo, args.pr_number, review)


if __name__ == "__main__":
    main()
