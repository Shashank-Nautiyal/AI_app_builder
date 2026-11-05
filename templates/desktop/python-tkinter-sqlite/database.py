from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("app.db")


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open'
            )
            """
        )


def list_items() -> list[sqlite3.Row]:
    with get_connection() as connection:
        return connection.execute("SELECT * FROM items ORDER BY id DESC").fetchall()


def add_item(title: str) -> None:
    with get_connection() as connection:
        connection.execute("INSERT INTO items (title) VALUES (?)", (title,))
        connection.commit()


def delete_item(item_id: int) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM items WHERE id = ?", (item_id,))
        connection.commit()
