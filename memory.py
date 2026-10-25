import os
import sqlite3
import time
import re
from pathlib import Path
from typing import List, Dict, Optional

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "memories.db"


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL UNIQUE,
            source TEXT DEFAULT 'explicit',
            created_at INTEGER NOT NULL,
            last_used_at INTEGER,
            use_count INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def add_memory(fact: str, source: str = "explicit") -> Dict[str, str]:
    if not fact:
        return {"status": "error", "message": "Empty fact"}
    fact = fact.strip()
    if len(fact) == 0:
        return {"status": "error", "message": "Empty fact"}
    if len(fact) > 500:
        fact = fact[:500].strip()
    
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO memories (fact, source, created_at) VALUES (?, ?, ?)",
            (fact, source, int(time.time())),
        )
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Got it, I'll remember that."}
    except sqlite3.IntegrityError:
        conn.close()
        return {"status": "exists", "message": "I already remember that."}
    except Exception as e:
        conn.close()
        return {"status": "error", "message": f"Error saving memory: {e}"}


STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "your", "can", "what", "who",
    "how", "why", "when", "where", "which", "that", "this", "with", "have", "has",
    "was", "were", "will", "would", "could", "should", "about", "does", "did",
    "tell", "please", "know", "from", "into", "his", "her", "its", "our", "his",
}


def _tokens(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def get_relevant_memories(query: str, limit: int = 5) -> List[Dict]:
    """Return memories sharing meaningful words with the query, best match first."""
    tokens = set(_tokens(query))
    if not tokens:
        return []
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, fact, use_count, created_at FROM memories").fetchall()
    conn.close()

    scored = []
    for row in rows:
        score = len(tokens & set(_tokens(row["fact"])))
        if score:
            scored.append((score, row["use_count"], row["created_at"], dict(row)))
    scored.sort(key=lambda x: x[:3], reverse=True)
    return [item[3] for item in scored[:limit]]


def list_memories(limit: int = 50) -> List[Dict]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, fact FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def forget(description: str) -> str:
    """Delete the single memory matching the description; refuse if ambiguous."""
    matches = get_relevant_memories(description, limit=5)
    if not matches:
        return "I couldn't find anything matching that in my memory."
    # Only treat as unambiguous if one memory clearly matches best
    top = len(set(_tokens(description)) & set(_tokens(matches[0]["fact"])))
    best = [m for m in matches
            if len(set(_tokens(description)) & set(_tokens(m["fact"]))) == top]
    if len(best) > 1:
        facts = "; ".join(m["fact"] for m in best)
        return f"That matches several memories: {facts}. Please be more specific."
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM memories WHERE id = ?", (best[0]["id"],))
    conn.commit()
    conn.close()
    return f"Okay, I've forgotten that {best[0]['fact']}."


def mark_used(memory_ids: List[int]):
    if not memory_ids:
        return
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for mid in memory_ids:
        cur.execute(
            "UPDATE memories SET last_used_at = ?, use_count = use_count + 1 WHERE id = ?",
            (int(time.time()), mid),
        )
    conn.commit()
    conn.close()


def extract_remember_fact(text: str) -> Optional[str]:
    if not text:
        return None
    text_clean = text.strip()
    # Case insensitive patterns
    # Whisper often adds a leading "please"/"hey," or a comma after the verb
    patterns = [
        r'^(?:hey,?\s+|please\s+)*remember(?:\s+that)?[,:]?\s+(.+)$',
        r'^(?:hey,?\s+|please\s+)*note\s+that[,:]?\s+(.+)$',
        r'^(?:hey,?\s+|please\s+)*keep\s+in\s+mind\s+that[,:]?\s+(.+)$',
    ]
    for pattern in patterns:
        m = re.match(pattern, text_clean, re.IGNORECASE)
        if m:
            fact = m.group(1).strip()
            # Remove trailing punctuation
            fact = fact.rstrip('.,!?;:')
            return fact if fact else None
    return None


def handle_memory_command(text: str) -> Optional[str]:
    """Handle explicit memory commands. Returns the reply, or None if not a memory command."""
    t = text.strip().rstrip(".,!?;:")
    fact = extract_remember_fact(t)
    if fact:
        return add_memory(fact)["message"]

    if re.match(r"^(?:hey,?\s+)?(?:what|tell me what)\s+do\s+you\s+(?:remember|know)\s+about\s+me$|"
                r"^(?:what|tell me what)\s+(?:do\s+you\s+)?(?:remember|know)$|"
                r"^what\s+(?:have\s+you\s+)?(?:remembered|stored)$|"
                r"^what\s+do\s+you\s+remember$", t, re.IGNORECASE):
        items = list_memories()
        if not items:
            return "I don't have anything saved yet."
        return "Here's what I remember. " + ". ".join(i["fact"] for i in items) + "."

    m = re.match(r"^(?:please\s+)?forget(?:\s+(?:that|about))?[,:]?\s+(.+)$", t, re.IGNORECASE)
    if m:
        return forget(m.group(1))
    return None
