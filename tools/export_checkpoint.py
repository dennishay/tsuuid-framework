#!/usr/bin/env python3
"""Export all 768 vectors to a checkpoint database for iPhone sync.

Usage:
    python3 tools/export_checkpoint.py [output_path]

Default output: ~/Dropbox/__768_sync/checkpoints/latest.db
"""
import sqlite3
import os
import sys


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Library/CloudStorage/Dropbox/__768_sync/checkpoints/latest.db"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        os.remove(out_path)

    conn_out = sqlite3.connect(out_path)
    conn_out.execute("PRAGMA journal_mode=WAL")
    conn_out.execute("""CREATE TABLE tsuuid_768 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE, title TEXT, vec BLOB,
        domain TEXT, version INTEGER DEFAULT 1, encoded_at TEXT
    )""")

    count = 0

    def insert(path, title, vec_blob, domain, date=""):
        nonlocal count
        if vec_blob and len(vec_blob) >= 1536:
            conn_out.execute(
                "INSERT OR IGNORE INTO tsuuid_768 (path,title,vec,domain,encoded_at) VALUES (?,?,?,?,?)",
                (path, (title or "")[:100], vec_blob[:1536], domain, date),
            )
            count += 1

    # CPC emails + docs
    cpc_db = os.path.expanduser("~/cpc/_automation/state.db")
    if os.path.exists(cpc_db):
        c = sqlite3.connect(cpc_db, timeout=10)
        for dts, subj, sender, date_sent, vec in c.execute(
            "SELECT dts, subject, sender, date_sent, vec_768 FROM emails WHERE vec_768 IS NOT NULL"
        ):
            insert(f"cpc:email:{dts}", subj, vec, "cpc:emails", date_sent)
        for fpath, title, doc_type, vec in c.execute(
            "SELECT file_path, title, doc_type, vec_768 FROM documents WHERE vec_768 IS NOT NULL"
        ):
            insert(f"cpc:doc:{fpath}", title or fpath, vec, "cpc:docs")
        c.close()

    # Dropbox files
    db = os.path.expanduser("~/.claude/dropbox_index.db")
    if os.path.exists(db):
        c = sqlite3.connect(db, timeout=10)
        for fpath, fname, source, vec in c.execute(
            "SELECT path, filename, source, vec_768 FROM files WHERE vec_768 IS NOT NULL"
        ):
            insert(f"dropbox:{fpath}", fname, vec, f"dropbox:{source or 'files'}")
        c.close()

    # Semantic docs + sessions
    home_db = os.path.expanduser("~/.claude/claude_home.db")
    if os.path.exists(home_db):
        c = sqlite3.connect(home_db, timeout=10)
        for uuid, vec, title, domain in c.execute("""
            SELECT sd.doc_uuid, sv.vec, sd.title, sd.domain
            FROM semantic_docs sd JOIN semantic_vectors sv ON sv.doc_uuid = sd.doc_uuid
            WHERE sv.model = 'labse' AND sv.vec IS NOT NULL
        """):
            insert(f"home:semantic:{uuid}", title, vec, domain or "home:semantic")
        for sess, turn, preview, vec, created in c.execute(
            "SELECT session_id, turn_number, text_preview, vec, created_at FROM session_turns WHERE vec IS NOT NULL"
        ):
            insert(f"home:session:{sess}:{turn}", preview, vec, "home:sessions", created)
        c.close()

    conn_out.commit()
    final = conn_out.execute("SELECT COUNT(*) FROM tsuuid_768").fetchone()[0]
    conn_out.close()
    size = os.path.getsize(out_path) / 1e6
    print(f"{final} vectors, {size:.0f} MB → {out_path}")


if __name__ == "__main__":
    main()
