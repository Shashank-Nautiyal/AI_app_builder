from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path("app.db")


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def load_records() -> pd.DataFrame:
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    return pd.DataFrame([dict(row) for row in rows])


init_db()
st.set_page_config(page_title="Streamlit App", layout="wide")
st.title("Streamlit App")

with st.form("record-form", clear_on_submit=True):
    title = st.text_input("Title")
    status = st.selectbox("Status", ["active", "pending", "done"])
    submitted = st.form_submit_button("Add")
    if submitted and title.strip():
        with get_connection() as connection:
            connection.execute("INSERT INTO records (title, status) VALUES (?, ?)", (title.strip(), status))
            connection.commit()
        st.success("Record added")

records = load_records()
if records.empty:
    st.info("No records yet")
else:
    st.dataframe(records, use_container_width=True, hide_index=True)
