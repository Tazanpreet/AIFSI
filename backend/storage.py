"""SQLite storage (standard library only). The database file lives in data/aifsi.db."""
import contextlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    text        TEXT NOT NULL,
    label       TEXT NOT NULL,
    title       TEXT NOT NULL,
    severity    TEXT NOT NULL,
    confidence  REAL NOT NULL,
    is_unknown  INTEGER NOT NULL DEFAULT 0,
    embedding   TEXT,
    model_id    TEXT,
    result_json TEXT NOT NULL,
    ai_summary  TEXT
);
CREATE TABLE IF NOT EXISTS log_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    name         TEXT NOT NULL,
    n_lines      INTEGER NOT NULL,
    summary_json TEXT NOT NULL
);
"""


def db_path() -> Path:
    return Path(os.environ.get("AIFSI_DB", ROOT / "data" / "aifsi.db"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextlib.contextmanager
def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


def _summary(row) -> dict:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "text": row["text"],
        "label": row["label"],
        "title": row["title"],
        "severity": row["severity"],
        "confidence": row["confidence"],
        "unknown": bool(row["is_unknown"]),
    }


def add_incident(text: str, result: dict, embedding=None, model_id=None) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO incidents (created_at, text, label, title, severity, confidence, is_unknown,"
            " embedding, model_id, result_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                _now(), text, result["label"], result["title"], result["severity"], float(result["confidence"]),
                1 if result.get("unknown") else 0,
                json.dumps(embedding) if embedding is not None else None,
                model_id, json.dumps(result),
            ),
        )
        return cur.lastrowid


def get_incident(incident_id: int):
    with connect() as c:
        row = c.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    if row is None:
        return None
    out = _summary(row)
    out["result"] = json.loads(row["result_json"])
    out["ai_summary"] = row["ai_summary"]
    return out


def list_incidents(limit: int = 20) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_summary(r) for r in rows]


def clear_incidents() -> int:
    with connect() as c:
        return c.execute("DELETE FROM incidents").rowcount


def set_summary(incident_id: int, text: str) -> bool:
    with connect() as c:
        return c.execute("UPDATE incidents SET ai_summary = ? WHERE id = ?", (text, incident_id)).rowcount > 0


def recent_embeddings(model_id: str, limit: int = 500) -> list[dict]:
    """Stored vectors from the SAME trained model (vectors from an older model are not comparable)."""
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM incidents WHERE embedding IS NOT NULL AND model_id = ? ORDER BY id DESC LIMIT ?",
            (model_id, limit),
        ).fetchall()
    out = []
    for r in rows:
        item = _summary(r)
        item["embedding"] = json.loads(r["embedding"])
        out.append(item)
    return out


def stats() -> dict:
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(timespec="seconds")
    with connect() as c:
        total = c.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        last_24h = c.execute("SELECT COUNT(*) FROM incidents WHERE created_at >= ?", (since,)).fetchone()[0]
        by_label = c.execute(
            "SELECT label, MAX(title) AS title, COUNT(*) AS n FROM incidents GROUP BY label ORDER BY n DESC"
        ).fetchall()
        by_sev = c.execute("SELECT severity, COUNT(*) AS n FROM incidents GROUP BY severity").fetchall()
        runs = c.execute("SELECT COUNT(*) FROM log_runs").fetchone()[0]
    return {
        "total": total,
        "last_24h": last_24h,
        "log_runs": runs,
        "by_label": [{"label": r["label"], "title": r["title"], "count": r["n"]} for r in by_label],
        "by_severity": {r["severity"]: r["n"] for r in by_sev},
    }


def add_log_run(name: str, n_lines: int, summary: dict) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO log_runs (created_at, name, n_lines, summary_json) VALUES (?,?,?,?)",
            (_now(), name, n_lines, json.dumps(summary)),
        )
        return cur.lastrowid


def list_log_runs(limit: int = 10) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM log_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [
        {"id": r["id"], "created_at": r["created_at"], "name": r["name"], "n_lines": r["n_lines"],
         "summary": json.loads(r["summary_json"])}
        for r in rows
    ]
