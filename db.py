"""
Database abstraction — SQLite (local dev) or PostgreSQL (production).

Set DATABASE_URL=postgresql://user:pass@host/dbname to use PostgreSQL.
Leave DATABASE_URL unset to use SQLite (path provided via db_connect).
"""
import os
import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("debug_coach")

DATABASE_URL = os.getenv("DATABASE_URL", "")
IS_POSTGRES = DATABASE_URL.startswith(("postgresql://", "postgres://"))

if IS_POSTGRES:
    import asyncpg  # type: ignore[import-not-found]
else:
    import aiosqlite  # type: ignore[import-not-found]

_pool = None  # asyncpg.Pool when IS_POSTGRES is True
_conn = None  # aiosqlite.Connection otherwise


def _to_pg(sql: str) -> str:
    """Replace ? placeholders with $1, $2, ... for PostgreSQL."""
    n = 0

    def _rep(_m: re.Match) -> str:
        nonlocal n
        n += 1
        return f"${n}"

    return re.sub(r"\?", _rep, sql)


# ── lifecycle ─────────────────────────────────────────────────────────────────


async def db_connect(sqlite_path: str) -> None:
    global _pool, _conn
    if IS_POSTGRES:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10, ssl="require")
        logger.info("PostgreSQL pool connected")
    else:
        _conn = await aiosqlite.connect(sqlite_path)
        _conn.row_factory = aiosqlite.Row
        await _conn.execute("PRAGMA journal_mode=WAL")
        await _conn.execute("PRAGMA foreign_keys=ON")
        logger.info("SQLite connected: %s", sqlite_path)


async def db_close() -> None:
    global _pool, _conn
    if IS_POSTGRES and _pool:
        await _pool.close()
        _pool = None
    elif _conn:
        await _conn.close()
        _conn = None


# ── schema ────────────────────────────────────────────────────────────────────

_SQLITE_DDL = """
    CREATE TABLE IF NOT EXISTS auth (
        user_id       TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        created_at    TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS tokens (
        token      TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL,
        token_type TEXT NOT NULL DEFAULT 'access',
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS users (
        user_id           TEXT PRIMARY KEY,
        session_id        TEXT NOT NULL,
        created_at        TEXT NOT NULL,
        total_queries     INTEGER DEFAULT 0,
        favorite_language TEXT,
        skill_level       TEXT,
        last_activity     TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS history (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id         TEXT NOT NULL,
        timestamp       TEXT NOT NULL,
        request_type    TEXT NOT NULL,
        language        TEXT NOT NULL,
        level           TEXT NOT NULL,
        query           TEXT,
        processing_time REAL,
        score           INTEGER
    );
    CREATE TABLE IF NOT EXISTS leaderboard (
        user_id                     TEXT PRIMARY KEY,
        total_score                 INTEGER DEFAULT 0,
        practice_sessions_completed INTEGER DEFAULT 0,
        last_activity               TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS practice_sessions (
        session_id            TEXT PRIMARY KEY,
        user_id               TEXT NOT NULL,
        language              TEXT NOT NULL,
        level                 TEXT NOT NULL,
        mode                  TEXT NOT NULL,
        questions             TEXT NOT NULL,
        answers               TEXT DEFAULT '{}',
        started_at            TEXT NOT NULL,
        completed_at          TEXT,
        total_score           INTEGER DEFAULT 0,
        total_possible_points INTEGER DEFAULT 0
    );
"""

_PG_DDL = [
    """CREATE TABLE IF NOT EXISTS auth (
        user_id       TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        created_at    TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS tokens (
        token      TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL,
        token_type TEXT NOT NULL DEFAULT 'access',
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS users (
        user_id           TEXT PRIMARY KEY,
        session_id        TEXT NOT NULL,
        created_at        TEXT NOT NULL,
        total_queries     INTEGER DEFAULT 0,
        favorite_language TEXT,
        skill_level       TEXT,
        last_activity     TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS history (
        id              BIGSERIAL PRIMARY KEY,
        user_id         TEXT NOT NULL,
        timestamp       TEXT NOT NULL,
        request_type    TEXT NOT NULL,
        language        TEXT NOT NULL,
        level           TEXT NOT NULL,
        query           TEXT,
        processing_time DOUBLE PRECISION,
        score           INTEGER
    )""",
    """CREATE TABLE IF NOT EXISTS leaderboard (
        user_id                     TEXT PRIMARY KEY,
        total_score                 INTEGER DEFAULT 0,
        practice_sessions_completed INTEGER DEFAULT 0,
        last_activity               TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS practice_sessions (
        session_id            TEXT PRIMARY KEY,
        user_id               TEXT NOT NULL,
        language              TEXT NOT NULL,
        level                 TEXT NOT NULL,
        mode                  TEXT NOT NULL,
        questions             TEXT NOT NULL,
        answers               TEXT DEFAULT '{}',
        started_at            TEXT NOT NULL,
        completed_at          TEXT,
        total_score           INTEGER DEFAULT 0,
        total_possible_points INTEGER DEFAULT 0
    )""",
]


async def init_db() -> None:
    if IS_POSTGRES:
        async with _pool.acquire() as conn:
            async with conn.transaction():
                for stmt in _PG_DDL:
                    await conn.execute(stmt)
        logger.info("PostgreSQL schema ready")
    else:
        await _conn.executescript(_SQLITE_DDL)
        await _conn.commit()
        # Migration: add token_type to existing SQLite databases
        try:
            await _conn.execute(
                "ALTER TABLE tokens ADD COLUMN token_type TEXT NOT NULL DEFAULT 'access'"
            )
            await _conn.commit()
        except Exception:
            pass
        logger.info("SQLite schema ready")


# ── write operations ──────────────────────────────────────────────────────────


async def db_execute(sql: str, *args: Any) -> None:
    if IS_POSTGRES:
        async with _pool.acquire() as conn:
            await conn.execute(_to_pg(sql), *args)
    else:
        await _conn.execute(sql, args)


async def db_commit() -> None:
    if not IS_POSTGRES and _conn:
        await _conn.commit()


# ── read operations  (always return plain dicts or None) ──────────────────────


async def db_fetchrow(sql: str, *args: Any) -> Optional[Dict[str, Any]]:
    if IS_POSTGRES:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(_to_pg(sql), *args)
            return dict(row) if row else None
    else:
        async with _conn.execute(sql, args) as cur:
            row = await cur.fetchone()
            return {k: row[k] for k in row.keys()} if row else None


async def db_fetchall(sql: str, *args: Any) -> List[Dict[str, Any]]:
    if IS_POSTGRES:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(_to_pg(sql), *args)
            return [dict(r) for r in rows]
    else:
        async with _conn.execute(sql, args) as cur:
            rows = await cur.fetchall()
            return [{k: row[k] for k in row.keys()} for row in rows]


async def db_fetchval(sql: str, *args: Any) -> Any:
    if IS_POSTGRES:
        async with _pool.acquire() as conn:
            return await conn.fetchval(_to_pg(sql), *args)
    else:
        async with _conn.execute(sql, args) as cur:
            row = await cur.fetchone()
            return row[0] if row else None
