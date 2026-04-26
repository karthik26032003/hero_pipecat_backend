import os
from datetime import datetime, date
import asyncpg
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

_pool: asyncpg.Pool | None = None


async def init_db():
    global _pool
    dsn = os.getenv("DATABASE_URL", "")
    _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10, ssl="require")
    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS venkanna_calls (
                call_uuid       TEXT PRIMARY KEY,
                customer_name   TEXT,
                from_number     TEXT,
                to_number       TEXT,
                status          TEXT NOT NULL DEFAULT 'pending',
                recording_url   TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                ended_at        TIMESTAMPTZ
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS venkanna_transcripts (
                id          SERIAL PRIMARY KEY,
                call_uuid   TEXT REFERENCES venkanna_calls(call_uuid) ON DELETE CASCADE,
                role        TEXT NOT NULL,
                text        TEXT NOT NULL,
                turn_index  INTEGER NOT NULL
            )
        """)
        await conn.execute("""
            ALTER TABLE venkanna_calls
            ADD COLUMN IF NOT EXISTS whatsapp_consent BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS whatsapp_sent    BOOLEAN DEFAULT FALSE
        """)
    logger.info("DB initialised — venkanna tables ready")


async def close_db():
    if _pool:
        await _pool.close()


def _row(r):
    if not r:
        return None
    out = {}
    for k, v in dict(r).items():
        if isinstance(v, (datetime, date)):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


async def insert_call(call_uuid: str, from_number: str, to_number: str, customer_name: str | None = None):
    async with _pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO venkanna_calls (call_uuid, from_number, to_number, customer_name)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (call_uuid) DO NOTHING
        """, call_uuid, from_number, to_number, customer_name)


async def update_call_ended(call_uuid: str):
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE venkanna_calls SET ended_at=NOW() WHERE call_uuid=$1", call_uuid
        )


async def update_call_status(call_uuid: str, status: str):
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE venkanna_calls SET status=$1 WHERE call_uuid=$2", status, call_uuid
        )


async def update_call_recording(call_uuid: str, recording_url: str):
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE venkanna_calls SET recording_url=$1 WHERE call_uuid=$2", recording_url, call_uuid
        )


async def insert_transcript(call_uuid: str, turns: list[dict]):
    async with _pool.acquire() as conn:
        await conn.executemany("""
            INSERT INTO venkanna_transcripts (call_uuid, role, text, turn_index)
            VALUES ($1, $2, $3, $4)
        """, [(call_uuid, t["role"], t["text"], t["index"]) for t in turns])


async def get_calls():
    async with _pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT *,
                CASE
                    WHEN ended_at IS NOT NULL
                    THEN EXTRACT(EPOCH FROM (ended_at - created_at))::INTEGER
                END AS duration_seconds
            FROM venkanna_calls
            ORDER BY created_at DESC
        """)
        return [_row(r) for r in rows]


async def get_call(call_uuid: str):
    async with _pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT *,
                CASE
                    WHEN ended_at IS NOT NULL
                    THEN EXTRACT(EPOCH FROM (ended_at - created_at))::INTEGER
                END AS duration_seconds
            FROM venkanna_calls
            WHERE call_uuid=$1
        """, call_uuid)
        return _row(row)


async def update_whatsapp_consent(call_uuid: str, consented: bool):
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE venkanna_calls SET whatsapp_consent=$1 WHERE call_uuid=$2", consented, call_uuid
        )


async def mark_whatsapp_sent(call_uuid: str):
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE venkanna_calls SET whatsapp_sent=TRUE WHERE call_uuid=$1", call_uuid
        )


async def get_call_by_phone(phone: str):
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM venkanna_calls WHERE to_number=$1 ORDER BY created_at DESC LIMIT 1", phone
        )
        return _row(row)


async def get_transcript(call_uuid: str):
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, text, turn_index FROM venkanna_transcripts WHERE call_uuid=$1 ORDER BY turn_index ASC",
            call_uuid
        )
        return [_row(r) for r in rows]
