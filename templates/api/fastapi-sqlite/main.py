from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from database import get_connection, init_db

app = FastAPI(title="FastAPI SQLite App")


class RecordCreate(BaseModel):
    title: str
    status: str = "active"


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/records")
def list_records() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    return [dict(row) for row in rows]


@app.post("/api/records")
def create_record(payload: RecordCreate) -> dict:
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO records (title, status) VALUES (?, ?)",
            (payload.title, payload.status),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM records WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row)


@app.delete("/api/records/{record_id}")
def delete_record(record_id: int) -> dict[str, str]:
    with get_connection() as connection:
        cursor = connection.execute("DELETE FROM records WHERE id = ?", (record_id,))
        connection.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"status": "deleted"}
