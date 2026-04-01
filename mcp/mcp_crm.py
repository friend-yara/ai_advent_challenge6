"""
mcp/mcp_crm.py — CRM domain module for the MCP router.

Public interface:
    CRM_TOOLS                              list[dict]  MCP tool definitions
    dispatch_crm_tool(name, args) -> dict              MCP result or raises
    init_crm()                                         init DB + seed data

Raises in dispatch_crm_tool:
    KeyError   — unknown tool name
    ValueError — invalid / missing arguments
"""

import json
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# DB path  (mcp/support.db — next to this file)
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).resolve().parent / "support.db"
_initialized = False

# ---------------------------------------------------------------------------
# Tool definitions (MCP-compatible schema)
# ---------------------------------------------------------------------------

CRM_TOOLS: list[dict] = [
    {
        "name": "get_ticket",
        "title": "Get support ticket",
        "description": (
            "Retrieve a support ticket by ID, including user profile data. "
            "Use when the user mentions a ticket number (e.g. T-001). "
            "Returns ticket details + user info (name, plan, platform)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "Ticket ID, e.g. 'T-001'",
                },
            },
            "required": ["ticket_id"],
        },
    },
    {
        "name": "search_tickets",
        "title": "Search tickets",
        "description": (
            "Search support tickets by text query and optional status filter. "
            "Returns matching tickets with user info."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text (matched against subject and description)",
                },
                "status": {
                    "type": "string",
                    "description": "Filter by status: open, in_progress, resolved, closed",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_user_tickets",
        "title": "List user tickets",
        "description": (
            "List all tickets for a specific user by user ID. "
            "Returns tickets sorted by creation date (newest first)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User ID, e.g. 'U-001'",
                },
            },
            "required": ["user_id"],
        },
    },
]

# ---------------------------------------------------------------------------
# Public initialiser
# ---------------------------------------------------------------------------

def init_crm():
    """Create tables and seed data if DB is empty. Idempotent."""
    global _initialized
    if _initialized:
        return
    _init_db()
    _initialized = True


# ---------------------------------------------------------------------------
# Public dispatch entry point
# ---------------------------------------------------------------------------

def dispatch_crm_tool(tool_name: str, arguments: dict) -> dict:
    """Dispatch a tools/call request to the appropriate CRM tool."""
    if tool_name == "get_ticket":
        ticket_id = (arguments.get("ticket_id") or "").strip()
        if not ticket_id:
            raise ValueError("Missing required: 'ticket_id'")
        return _get_ticket(ticket_id)

    if tool_name == "search_tickets":
        query = (arguments.get("query") or "").strip()
        if not query:
            raise ValueError("Missing required: 'query'")
        status = (arguments.get("status") or "").strip() or None
        return _search_tickets(query, status)

    if tool_name == "list_user_tickets":
        user_id = (arguments.get("user_id") or "").strip()
        if not user_id:
            raise ValueError("Missing required: 'user_id'")
        return _list_user_tickets(user_id)

    raise KeyError(f"Unknown CRM tool: {tool_name!r}")


# ---------------------------------------------------------------------------
# Internal: DB init + seed
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """Get a connection with row_factory for dict-like access."""
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db():
    """Create tables and seed with sample data."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            plan TEXT NOT NULL DEFAULT 'free',
            platform TEXT NOT NULL DEFAULT 'linux',
            registered TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            subject TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            priority TEXT NOT NULL DEFAULT 'medium',
            category TEXT NOT NULL DEFAULT 'general',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)

    # Seed only if tables are empty
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        _seed_data(conn)

    conn.commit()
    conn.close()


def _seed_data(conn: sqlite3.Connection):
    """Insert sample users and tickets."""
    users = [
        ("U-001", "Алексей Петров", "alexey@example.com", "free", "linux", "2025-11-15"),
        ("U-002", "Мария Иванова", "maria@example.com", "pro", "macos", "2025-12-01"),
        ("U-003", "Дмитрий Козлов", "dmitry@example.com", "enterprise", "linux", "2026-01-10"),
        ("U-004", "Анна Смирнова", "anna@example.com", "free", "windows", "2026-02-20"),
    ]
    conn.executemany(
        "INSERT INTO users (id, name, email, plan, platform, registered) VALUES (?,?,?,?,?,?)",
        users,
    )

    tickets = [
        ("T-001", "U-001", "Высокая стоимость API запросов",
         "Каждый запрос стоит ~$0.05, за день набегает $5+. "
         "Использую gpt-4.1 для всех задач. Как снизить расходы?",
         "open", "medium", "billing", "2026-03-20", "2026-03-20"),

        ("T-002", "U-001", "Не работает авторизация OpenAI",
         "После обновления API key перестал работать. "
         "Получаю ошибку 'invalid_api_key'. Ключ точно правильный, "
         "проверял в dashboard.",
         "open", "high", "auth", "2026-03-25", "2026-03-25"),

        ("T-003", "U-002", "MCP инструменты не подключаются",
         "При запуске CLI не вижу MCP tools в /tool list. "
         "Раньше работало, после перезагрузки сервера перестало. "
         "Weather и git tools пропали.",
         "in_progress", "high", "mcp", "2026-03-22", "2026-03-26"),

        ("T-004", "U-002", "RAG выдаёт нерелевантные результаты",
         "Спрашиваю про Python code style, а получаю куски из Java guide. "
         "Индексировал docs/ через fixed chunker. "
         "Может structured chunker будет лучше?",
         "open", "medium", "rag", "2026-03-27", "2026-03-27"),

        ("T-005", "U-003", "Как настроить инварианты для нового проекта?",
         "Создал новый профиль profiles/myproject/, но не понимаю "
         "формат INVARIANTS.yaml. Какие секции обязательны? "
         "Как добавить свои banned правила?",
         "open", "low", "cli", "2026-03-28", "2026-03-28"),

        ("T-006", "U-003", "Субагент research не вызывается автоматически",
         "Ожидаю что planner сам вызовет delegate_research, "
         "но он пытается ответить сам. Как заставить его делегировать?",
         "open", "medium", "cli", "2026-03-29", "2026-03-29"),

        ("T-007", "U-004", "Ошибка при /state EXEC",
         "Пишу /state EXEC и получаю 'план пуст'. "
         "Но я только что попросил planner создать план и он ответил. "
         "Почему план не сохранился?",
         "resolved", "medium", "cli", "2026-03-15", "2026-03-18"),

        ("T-008", "U-004", "Как использовать Ollama вместо OpenAI?",
         "Хочу запускать модели локально чтобы не платить за API. "
         "Установила Ollama, но не знаю какие флаги указать в CLI. "
         "Какие модели поддерживаются?",
         "open", "low", "cli", "2026-03-30", "2026-03-30"),

        ("T-009", "U-001", "Reminder pipeline не выполняется",
         "Создал reminder с pipeline из get_forecast + save_to_file. "
         "Reminder сработал, но pipeline не запустился. "
         "В логах ничего нет.",
         "open", "high", "mcp", "2026-03-31", "2026-03-31"),
    ]
    conn.executemany(
        "INSERT INTO tickets (id, user_id, subject, description, status, "
        "priority, category, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        tickets,
    )


# ---------------------------------------------------------------------------
# Internal: tool implementations
# ---------------------------------------------------------------------------

def _format_ticket(row: sqlite3.Row) -> dict:
    """Format a ticket row into a readable dict."""
    return {
        "id": row["id"],
        "subject": row["subject"],
        "description": row["description"],
        "status": row["status"],
        "priority": row["priority"],
        "category": row["category"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _format_user(row: sqlite3.Row) -> dict:
    """Format a user row into a readable dict."""
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "plan": row["plan"],
        "platform": row["platform"],
        "registered": row["registered"],
    }


def _get_ticket(ticket_id: str) -> dict:
    """Get ticket by ID with user info."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT t.*, u.name as user_name, u.email as user_email, "
        "u.plan as user_plan, u.platform as user_platform, u.registered as user_registered "
        "FROM tickets t JOIN users u ON t.user_id = u.id "
        "WHERE t.id = ?",
        (ticket_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return {
            "content": [{"type": "text", "text": f"Тикет '{ticket_id}' не найден."}],
            "data": {},
        }

    ticket = {
        "id": row["id"],
        "subject": row["subject"],
        "description": row["description"],
        "status": row["status"],
        "priority": row["priority"],
        "category": row["category"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    user = {
        "id": row["user_id"],
        "name": row["user_name"],
        "email": row["user_email"],
        "plan": row["user_plan"],
        "platform": row["user_platform"],
        "registered": row["user_registered"],
    }

    text = (
        f"Тикет {ticket['id']}: {ticket['subject']}\n"
        f"Статус: {ticket['status']} | Приоритет: {ticket['priority']} | "
        f"Категория: {ticket['category']}\n"
        f"Описание: {ticket['description']}\n"
        f"Создан: {ticket['created_at']}\n\n"
        f"Пользователь: {user['name']} ({user['email']})\n"
        f"План: {user['plan']} | Платформа: {user['platform']} | "
        f"Зарегистрирован: {user['registered']}"
    )

    return {
        "content": [{"type": "text", "text": text}],
        "data": {"ticket": ticket, "user": user},
    }


def _search_tickets(query: str, status: str | None = None) -> dict:
    """Search tickets by text, optionally filtered by status."""
    conn = _get_conn()
    like = f"%{query}%"

    if status:
        rows = conn.execute(
            "SELECT t.*, u.name as user_name, u.plan as user_plan "
            "FROM tickets t JOIN users u ON t.user_id = u.id "
            "WHERE (t.subject LIKE ? OR t.description LIKE ?) AND t.status = ? "
            "ORDER BY t.created_at DESC",
            (like, like, status),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT t.*, u.name as user_name, u.plan as user_plan "
            "FROM tickets t JOIN users u ON t.user_id = u.id "
            "WHERE t.subject LIKE ? OR t.description LIKE ? "
            "ORDER BY t.created_at DESC",
            (like, like),
        ).fetchall()
    conn.close()

    if not rows:
        return {
            "content": [{"type": "text", "text": f"Тикеты по запросу '{query}' не найдены."}],
            "data": {"tickets": [], "count": 0},
        }

    lines = [f"Найдено тикетов: {len(rows)}\n"]
    tickets = []
    for r in rows:
        lines.append(
            f"  {r['id']}: {r['subject']} [{r['status']}] "
            f"(пользователь: {r['user_name']}, план: {r['user_plan']})"
        )
        tickets.append({
            "id": r["id"], "subject": r["subject"],
            "status": r["status"], "priority": r["priority"],
            "user_name": r["user_name"], "user_plan": r["user_plan"],
        })

    return {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "data": {"tickets": tickets, "count": len(tickets)},
    }


def _list_user_tickets(user_id: str) -> dict:
    """List all tickets for a user."""
    conn = _get_conn()

    user_row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if user_row is None:
        conn.close()
        return {
            "content": [{"type": "text", "text": f"Пользователь '{user_id}' не найден."}],
            "data": {},
        }

    rows = conn.execute(
        "SELECT * FROM tickets WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()

    user = _format_user(user_row)
    lines = [
        f"Пользователь: {user['name']} ({user['plan']}, {user['platform']})",
        f"Тикетов: {len(rows)}\n",
    ]
    tickets = []
    for r in rows:
        t = _format_ticket(r)
        lines.append(f"  {t['id']}: {t['subject']} [{t['status']}, {t['priority']}]")
        tickets.append(t)

    return {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "data": {"user": user, "tickets": tickets, "count": len(tickets)},
    }
