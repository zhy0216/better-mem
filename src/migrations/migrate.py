import asyncio
import os
from pathlib import Path

import asyncpg

from src.config import settings

MIGRATIONS_DIR = Path(__file__).parent


async def run_migrations() -> None:
    conn = await asyncpg.connect(dsn=settings.DATABASE_URL)
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)

        applied = {r["name"] for r in await conn.fetch("SELECT name FROM _migrations ORDER BY id")}

        sql_files = sorted(f for f in MIGRATIONS_DIR.glob("*.sql"))
        for sql_file in sql_files:
            name = sql_file.name
            if name in applied:
                print(f"  skip {name} (already applied)")
                continue
            print(f"  apply {name} ...")
            sql = sql_file.read_text()
            await conn.execute(sql)
            await conn.execute("INSERT INTO _migrations (name) VALUES ($1)", name)
            print(f"  done  {name}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_migrations())
