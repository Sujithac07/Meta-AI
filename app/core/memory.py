"""
Persistent conversation memory using SQLite.
Supports multiple named sessions per user.
"""
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from app.providers.base import ChatMessage
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Ensure the data directory exists
DB_DIR = Path("data")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "sessions.db"

class ConversationMemory:
    
    def __init__(self, session_id: str, max_history: int = 20):
        self.session_id = session_id
        self.max_history = max_history
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)"
        )
        conn.commit()
        conn.close()
    
    def add_message(self, role: str, content: str, metadata: dict = None):
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO messages VALUES (NULL,?,?,?,?,?)",
            (
                self.session_id, role, content,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(metadata or {})
            )
        )
        conn.commit()
        conn.close()
        logger.debug(f"Saved {role} message to session {self.session_id}")
    
    def get_history(self) -> List[ChatMessage]:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            """SELECT role, content FROM messages 
               WHERE session_id=? 
               ORDER BY id DESC LIMIT ?""",
            (self.session_id, self.max_history)
        ).fetchall()
        conn.close()
        # Return in chronological order
        return [ChatMessage(role=r, content=c) for r, c in reversed(rows)]
    
    def clear(self):
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "DELETE FROM messages WHERE session_id=?", 
            (self.session_id,)
        )
        conn.commit()
        conn.close()
    
    def get_summary_stats(self) -> dict:
        conn = sqlite3.connect(DB_PATH)
        stats = conn.execute(
            """SELECT COUNT(*), MIN(timestamp), MAX(timestamp) 
               FROM messages WHERE session_id=?""",
            (self.session_id,)
        ).fetchone()
        conn.close()
        return {
            "total_messages": stats[0] if stats else 0,
            "first_message": stats[1] if stats else None,
            "last_message": stats[2] if stats else None
        }
