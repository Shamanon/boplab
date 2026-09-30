# modules/memory.py
import sqlite3
import time
import config

def get_db():
    return sqlite3.connect(config.DB_PATH)

def save_memory(key, value):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO memories (key, value) VALUES (?, ?)",
        (key.lower(), value),
    )
    conn.commit()
    conn.close()

def recall_memories():
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT key, value FROM memories")
        rows = c.fetchall()
        conn.close()
        if not rows:
            return "No stored memories yet."
        return "\n".join([f"- {k}: {v}" for k, v in rows])
    except Exception:
        return "No memory database initialized."

def recall_lore(user_msg=""):
    try:
        conn = get_db()
        c = conn.cursor()
        keywords = [w for w in user_msg.lower().split() if len(w) > 4]
        if keywords:
            query = keywords[0]
            c.execute(
                "SELECT chunk FROM lore WHERE chunk LIKE ? ORDER BY RANDOM() LIMIT 1",
                (f"%{query}%",),
            )
            row = c.fetchone()
            if row:
                conn.close()
                return row[0]

        c.execute("SELECT chunk FROM lore ORDER BY RANDOM() LIMIT 1")
        row = c.fetchone()
        conn.close()
        return row[0] if row else "Free boppers gotta stick together."
    except Exception:
        return "Free boppers gotta stick together."

def log_history(role, content):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute(
            "INSERT INTO chat_history (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, time.time())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] History Logging Error: {e}")

def recall_recent_history(limit=6):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute(
            "SELECT role, content FROM chat_history ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = c.fetchall()
        conn.close()
        rows.reverse()
        return "\n".join([f"{'Joshua' if r == 'user' else 'BABS'}: {c}" for r, c in rows])
    except Exception:
        return ""
