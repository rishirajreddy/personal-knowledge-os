"""Personal knowledge base: SQLite full-text search over folders you configure.

Usage:  python knowledge.py add ~/Notes     # register a folder
        python knowledge.py remove ~/Notes
        python knowledge.py sources
        python knowledge.py index           # (re)index changed files
        python knowledge.py search "query"
"""
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

from memory import DATA_DIR, STOPWORDS

DB_PATH = DATA_DIR / "knowledge.db"
SOURCES_PATH = DATA_DIR / "sources.json"

EXTENSIONS = {".md", ".txt", ".rst"}
MAX_FILE_BYTES = 1_000_000
CHUNK_CHARS = 800


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL)")
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
        "USING fts5(path UNINDEXED, heading UNINDEXED, body, tokenize='porter')"
    )
    return conn


# ---- sources -------------------------------------------------------------

def get_sources() -> List[str]:
    try:
        return json.loads(SOURCES_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_sources(sources: List[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_PATH.write_text(json.dumps(sorted(set(sources)), indent=2))


def add_source(folder: str) -> str:
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        return f"{path} is not a folder."
    _save_sources(get_sources() + [str(path)])
    return f"Added {path}. Run index to read it."


def remove_source(folder: str) -> str:
    path = str(Path(folder).expanduser().resolve())
    _save_sources([s for s in get_sources() if s != path])
    return f"Removed {path}. Run index to drop its content."


# ---- indexing ------------------------------------------------------------

def _chunk(text: str) -> List[Dict[str, str]]:
    """Split on paragraphs, tracking the latest markdown heading, ~CHUNK_CHARS each."""
    chunks, heading, buf = [], "", ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        m = re.match(r"#+\s+(.*)", para)
        if m:
            heading = m.group(1).strip()
        if buf and len(buf) + len(para) > CHUNK_CHARS:
            chunks.append({"heading": heading, "body": buf})
            buf = ""
        buf = f"{buf}\n\n{para}".strip()
    if buf:
        chunks.append({"heading": heading, "body": buf})
    return chunks


def _iter_files(root: Path):
    for p in root.rglob("*"):
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue  # hidden files and folders
        if p.is_file() and p.suffix.lower() in EXTENSIONS and p.stat().st_size <= MAX_FILE_BYTES:
            yield p


def index_sources() -> Dict[str, int]:
    """Index new/changed files and drop deleted ones. Returns counts."""
    conn = _connect()
    known = dict(conn.execute("SELECT path, mtime FROM files"))
    seen, added, removed = set(), 0, 0
    for src in get_sources():
        root = Path(src)
        if not root.is_dir():
            continue
        for p in _iter_files(root):
            path, mtime = str(p), p.stat().st_mtime
            seen.add(path)
            if known.get(path) == mtime:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
            conn.executemany(
                "INSERT INTO chunks (path, heading, body) VALUES (?, ?, ?)",
                [(path, c["heading"], c["body"]) for c in _chunk(text)],
            )
            conn.execute("INSERT OR REPLACE INTO files VALUES (?, ?)", (path, mtime))
            added += 1
    for path in set(known) - seen:
        conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
        removed += 1
    conn.commit()
    conn.close()
    return {"indexed": added, "removed": removed, "files": len(seen)}


# ---- search --------------------------------------------------------------

# Common question words that say nothing about a topic
GENERIC_WORDS = {
    "difference", "differences", "between", "versus", "explain", "describe", "compare",
    "example", "examples", "give", "make", "use", "using", "used", "work", "works",
    "good", "best", "better", "thing", "things", "mean", "means", "meaning", "learn",
    "programming", "code", "coding", "write", "help", "need", "want", "like", "just",
}


def asks_about_notes(query: str) -> bool:
    """True if the user explicitly refers to their own notes/documents."""
    return bool(re.search(r"\b(?:my|the)\s+(?:notes?|documents?|docs|files?|knowledge base)\b|"
                          r"\b(?:according to|in|from)\s+(?:my\s+)?notes\b", query, re.IGNORECASE))


def search(query: str, limit: int = 3, explicit: bool = False) -> List[Dict[str, str]]:
    """Best-matching chunks. For ordinary questions, at least two distinctive query
    terms must appear in the same chunk, so general questions don't drag in notes.
    If the user explicitly asks about their notes (explicit=True), one term is enough."""
    terms = [w for w in re.findall(r"[a-z0-9]+", query.lower())
             if len(w) > 2 and w not in STOPWORDS and w not in GENERIC_WORDS]
    if not terms or not DB_PATH.exists():
        return []
    need = 1 if explicit else 2
    conn = _connect()
    fts_query = " OR ".join(f'"{t}"' for t in terms)
    rows = conn.execute(
        "SELECT path, heading, body FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT 20",
        (fts_query,),
    ).fetchall()
    conn.close()

    results = []
    for path, heading, body in rows:
        lowered = body.lower() + " " + heading.lower()
        # crude stem match: a term counts if its first 5 letters appear
        hits = sum(1 for t in terms if t[:5] in lowered)
        if hits >= need:
            results.append({"path": path, "heading": heading, "body": body})
        if len(results) >= limit:
            break
    return results


def format_snippets(results: List[Dict[str, str]]) -> List[str]:
    return [f"[{Path(r['path']).name}{' > ' + r['heading'] if r['heading'] else ''}] {r['body']}"
            for r in results]


def handle_knowledge_command(text: str) -> Optional[str]:
    t = text.strip().rstrip(".,!?;:")
    if re.match(r"^(?:please\s+)?(?:re-?index|refresh|update)\s+(?:my\s+)?(?:notes|knowledge|documents)(?:\s+index)?$",
                t, re.IGNORECASE):
        if not get_sources():
            return "No note folders are configured yet. Add one with python knowledge.py add and a folder path."
        r = index_sources()
        return f"Done. I indexed {r['indexed']} new or changed files, out of {r['files']} total."
    return None


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "add" and args:
        print(add_source(args[0]))
    elif cmd == "remove" and args:
        print(remove_source(args[0]))
    elif cmd == "sources":
        print("\n".join(get_sources()) or "No sources configured.")
    elif cmd == "index":
        print(index_sources())
    elif cmd == "search" and args:
        for r in search(" ".join(args)):
            print(f"--- {r['path']} ({r['heading']})\n{r['body']}\n")
    else:
        print(__doc__)
