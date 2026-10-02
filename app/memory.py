import json
import sqlite3
from pathlib import Path

from app.catalog import PRIVACY


class MemoryStore:
    """Local prototype memory. user_id is a namespace, NOT authentication."""

    def __init__(self, path):
        self.path = path

    def connect(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=1)
        db.execute(
            "CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, payload TEXT NOT NULL, privacy TEXT NOT NULL)"
        )
        db.execute("CREATE INDEX IF NOT EXISTS memory_user ON memories(user_id, id)")
        return db

    def read(self, user_id):
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload, privacy FROM memories WHERE user_id=? ORDER BY id DESC LIMIT 3",
                (user_id,),
            ).fetchall()
        # Preserve chronological order, mark newest facts as taking precedence in prompt.
        rows.reverse()
        messages = [m for payload, _ in rows for m in json.loads(payload)]
        privacy = max((p for _, p in rows), key=PRIVACY.get, default="public")
        return messages, privacy

    def write(self, user_id, messages, privacy):
        with self.connect() as db:
            db.execute(
                "INSERT INTO memories(user_id,payload,privacy) VALUES (?,?,?)",
                (user_id, json.dumps(messages), privacy),
            )
