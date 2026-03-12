"""
mcp/mcp_scheduler.py — Scheduler domain module for the MCP router.

Public interface:
    SCHEDULER_TOOLS                              list[dict]  MCP tool definitions
    dispatch_scheduler_tool(name, args) -> dict              MCP result or raises
    init_scheduler()                                         init DB + start background loop

Raises in dispatch_scheduler_tool:
    KeyError   — unknown tool name
    ValueError — invalid / missing arguments
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# DB path  (mcp/reminders.db — next to this file, survives restarts)
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).resolve().parent / "reminders.db"

_POLL_INTERVAL = 2   # seconds between background loop ticks
_initialized = False  # guard: init_scheduler() is idempotent

# ---------------------------------------------------------------------------
# Tool definitions (MCP-compatible schema)
# ---------------------------------------------------------------------------

SCHEDULER_TOOLS: list[dict] = [
    {
        "name": "reminder",
        "title": "Reminder",
        "description": (
            "Create a delayed reminder or check an existing reminder's status. "
            "Use this tool whenever the user asks to be reminded about something. "
            "To CREATE: provide 'text' (what to remind) and one of the delay params "
            "(delay_seconds, delay_minutes, delay_hours, delay_days). "
            "To CHECK STATUS: provide 'job_id' returned from a previous call. "
            "Returns job_id, status (scheduled/completed), and a human-readable summary."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "What to remind the user about",
                },
                "delay_seconds": {
                    "type": "integer",
                    "description": "Задержка в секундах (default: 30)",
                    "default": 30,
                },
                "delay_minutes": {
                    "type": "number",
                    "description": "Задержка в минутах",
                },
                "delay_hours": {
                    "type": "number",
                    "description": "Задержка в часах",
                },
                "delay_days": {
                    "type": "number",
                    "description": "Задержка в днях",
                },
                "job_id": {
                    "type": "string",
                    "description": "Job ID of an existing reminder to check its status",
                },
                "pipeline": {
                    "type": "array",
                    "description": (
                        "Ordered list of MCP tool calls to execute when the reminder fires. "
                        "Each step: {\"tool\": \"tool_name\", \"args\": {...}}. "
                        "Use \"{prev_output}\" in args to inject text output of the previous step. "
                        "Example: [{\"tool\": \"get_forecast\", \"args\": {\"place\": \"London\", \"days\": 3}}, "
                        "{\"tool\": \"summarize_forecast\", \"args\": {\"place\": \"London\", \"days\": 3}}, "
                        "{\"tool\": \"save_to_file\", \"args\": {\"content\": \"{prev_output}\", \"filename\": \"forecast.txt\"}}]"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "args": {"type": "object"},
                        },
                        "required": ["tool", "args"],
                    },
                },
            },
            "required": [],
        },
    }
]

# ---------------------------------------------------------------------------
# Public initialiser
# ---------------------------------------------------------------------------

def init_scheduler():
    """
    Create the reminders table (if absent) and start the background loop.
    Safe to call multiple times — only initialises once per process.
    """
    global _initialized
    if _initialized:
        return
    _init_db()
    t = threading.Thread(target=_background_loop, daemon=True, name="scheduler-loop")
    t.start()
    _initialized = True


# ---------------------------------------------------------------------------
# Public dispatch entry point
# ---------------------------------------------------------------------------

def dispatch_scheduler_tool(tool_name: str, arguments: dict) -> dict:
    """
    Dispatch a tools/call request to the appropriate scheduler tool.

    Returns an MCP-compatible result dict:
        {"content": [{"type": "text", "text": "..."}], "data": {...}}

    Raises:
        KeyError    if tool_name is not a known scheduler tool
        ValueError  if required arguments are missing or invalid
    """
    if tool_name != "reminder":
        raise KeyError(f"Unknown scheduler tool: {tool_name!r}")

    text     = (arguments.get("text") or "").strip()
    job_id   = (arguments.get("job_id") or "").strip()

    if text:
        delay_s, human_label = _parse_delay(arguments)
        pipeline = arguments.get("pipeline")  # list | None
        return _create_reminder(text, delay_s, human_label, pipeline=pipeline)
    elif job_id:
        return _check_reminder(job_id)
    else:
        raise ValueError(
            "Provide 'text' (and optionally 'delay_seconds') to create a reminder, "
            "or 'job_id' to check an existing reminder's status."
        )


# ---------------------------------------------------------------------------
# Internal: delay parsing helpers
# ---------------------------------------------------------------------------

def _fmt_num(v) -> int | float:
    """Return int if value is whole number, else float."""
    f = float(v)
    return int(f) if f == int(f) else f


def _parse_delay(arguments: dict) -> tuple[int, str]:
    """Parse delay from arguments dict. Returns (delay_seconds, human_label)."""
    if (v := arguments.get("delay_days")) is not None:
        secs = int(float(v) * 86400)
        label = f"{_fmt_num(v)} дн."
    elif (v := arguments.get("delay_hours")) is not None:
        secs = int(float(v) * 3600)
        label = f"{_fmt_num(v)} ч."
    elif (v := arguments.get("delay_minutes")) is not None:
        secs = int(float(v) * 60)
        label = f"{_fmt_num(v)} мин."
    else:
        v = arguments.get("delay_seconds", 30)
        secs = int(v)
        label = f"{secs} сек."
    if not (1 <= secs <= 86400):
        raise ValueError("Задержка должна быть от 1 секунды до 24 часов")
    return secs, label


# ---------------------------------------------------------------------------
# Internal: create reminder
# ---------------------------------------------------------------------------

def _create_reminder(text: str, delay_seconds: int, human_delay: str,
                     pipeline: list | None = None) -> dict:
    """Insert a new reminder and return a scheduled-status MCP result."""
    job_id       = uuid.uuid4().hex[:12]
    now          = _utcnow()
    due_at       = now + timedelta(seconds=delay_seconds)
    created_at   = _fmt(now)
    due_at_str   = _fmt(due_at)
    pipeline_json = json.dumps(pipeline, ensure_ascii=False) if pipeline else None

    with _connect() as conn:
        conn.execute(
            "INSERT INTO reminders "
            "(job_id, text, delay_seconds, pipeline, status, created_at, due_at, completed_at) "
            "VALUES (?, ?, ?, ?, 'scheduled', ?, ?, NULL)",
            (job_id, text, delay_seconds, pipeline_json, created_at, due_at_str),
        )

    summary = f"reminder: «{text}» через {human_delay}"
    return {
        "content": [{"type": "text", "text": summary}],
        "data": {
            "job_id": job_id,
            "status": "scheduled",
            "text": text,
            "delay_seconds": delay_seconds,
            "pipeline": pipeline,
            "created_at": created_at,
            "due_at": due_at_str,
        },
    }


# ---------------------------------------------------------------------------
# Internal: check reminder status
# ---------------------------------------------------------------------------

def _check_reminder(job_id: str) -> dict:
    """Return current status or aggregated result for an existing reminder."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT job_id, text, delay_seconds, pipeline, status, "
            "       created_at, due_at, completed_at "
            "FROM reminders WHERE job_id = ?",
            (job_id,),
        ).fetchone()

    if row is None:
        raise ValueError(f"Reminder not found: {job_id!r}")

    r_job_id, r_text, r_delay, r_pipeline_json, r_status, r_created, r_due, r_completed = row
    r_pipeline = json.loads(r_pipeline_json) if r_pipeline_json else None

    if r_status == "completed":
        created_dt  = _parse(r_created)
        completed_dt = _parse(r_completed)
        actual_s    = int((completed_dt - created_dt).total_seconds())
        summary = (
            f"Reminder «{r_text}» выполнен. "
            f"Плановая задержка: {r_delay} сек. "
            f"Фактическое выполнение: {actual_s} сек."
        )
        return {
            "content": [{"type": "text", "text": summary}],
            "data": {
                "job_id": r_job_id,
                "status": "completed",
                "text": r_text,
                "delay_seconds": r_delay,
                "pipeline": r_pipeline,
                "created_at": r_created,
                "due_at": r_due,
                "completed_at": r_completed,
                "actual_delay_seconds": actual_s,
                "summary": summary,
            },
        }
    else:
        # still scheduled — calculate remaining time
        due_dt    = _parse(r_due)
        now       = _utcnow()
        remaining = max(0, int((due_dt - now).total_seconds()))
        summary = (
            f"Reminder «{r_text}» ещё не выполнен. "
            f"Осталось ≈{remaining} сек (до {r_due} UTC)."
        )
        return {
            "content": [{"type": "text", "text": summary}],
            "data": {
                "job_id": r_job_id,
                "status": "scheduled",
                "text": r_text,
                "delay_seconds": r_delay,
                "pipeline": r_pipeline,
                "created_at": r_created,
                "due_at": r_due,
                "seconds_remaining": remaining,
            },
        }


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------

def _background_loop():
    """Daemon thread: marks reminders as completed when their due_at has passed."""
    import time
    while True:
        try:
            _tick()
        except Exception:
            pass  # never let the loop die on transient errors
        time.sleep(_POLL_INTERVAL)


def _tick():
    """Single poll cycle: complete all overdue scheduled reminders."""
    now_str = _fmt(_utcnow())
    with _connect() as conn:
        overdue = conn.execute(
            "SELECT job_id FROM reminders "
            "WHERE status = 'scheduled' AND due_at <= ?",
            (now_str,),
        ).fetchall()
        for (job_id,) in overdue:
            conn.execute(
                "UPDATE reminders SET status = 'completed', completed_at = ? "
                "WHERE job_id = ? AND status = 'scheduled'",
                (now_str, job_id),
            )


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _init_db():
    """Create the reminders table if it does not exist; migrate existing DBs."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                job_id        TEXT PRIMARY KEY,
                text          TEXT NOT NULL,
                delay_seconds INTEGER NOT NULL,
                pipeline      TEXT DEFAULT NULL,
                status        TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                due_at        TEXT NOT NULL,
                completed_at  TEXT
            )
        """)
        # Migrate existing DBs that lack the pipeline column
        existing = {row[1] for row in conn.execute("PRAGMA table_info(reminders)")}
        if "pipeline" not in existing:
            conn.execute(
                "ALTER TABLE reminders ADD COLUMN pipeline TEXT DEFAULT NULL"
            )


def _connect() -> sqlite3.Connection:
    """Open a SQLite connection (thread-safe, auto-commit via context manager)."""
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Datetime helpers (UTC, no tzinfo dependency)
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
