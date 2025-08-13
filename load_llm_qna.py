#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Q&A → SQLite loader for multi-LLM comparison sheets.

- Accepts .xlsx/.xls/.csv
- Handles multi-sheet Excel (pick by name or index)
- Assumes first column = question text; remaining columns = model names
- Reshapes wide → long (one row per question–model)
- Creates schema (questions/models/answers/themes/question_themes + FTS)
- Idempotent: re-running updates answers without duplicating
"""

import argparse
import csv
import datetime as dt
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


# --------------------------- Schema ---------------------------------

SCHEMA_SQL = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS questions (
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL UNIQUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS answers (
  id INTEGER PRIMARY KEY,
  question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
  model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  answer_text TEXT NOT NULL,
  tokens_used INTEGER,
  latency_ms INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(question_id, model_id)
);

CREATE TABLE IF NOT EXISTS themes (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS question_themes (
  question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
  theme_id INTEGER NOT NULL REFERENCES themes(id) ON DELETE CASCADE,
  PRIMARY KEY (question_id, theme_id)
);

CREATE INDEX IF NOT EXISTS idx_answers_question ON answers(question_id);
CREATE INDEX IF NOT EXISTS idx_answers_model ON answers(model_id);
CREATE INDEX IF NOT EXISTS idx_qthemes_theme ON question_themes(theme_id);
"""

# Optional FTS on questions.text (nice for later search)
FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS questions_fts
  USING fts5(text, content='questions', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS questions_ai AFTER INSERT ON questions BEGIN
  INSERT INTO questions_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS questions_ad AFTER DELETE ON questions BEGIN
  INSERT INTO questions_fts(questions_fts, rowid, text)
  VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS questions_au AFTER UPDATE ON questions BEGIN
  INSERT INTO questions_fts(questions_fts, rowid, text)
  VALUES ('delete', old.id, old.text);
  INSERT INTO questions_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


# --------------------------- Helpers --------------------------------

def clean_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    # Normalize whitespace and strip weird control chars
    s = s.replace("\r", "\n").replace("\u00a0", " ")
    s = "\n".join(line.rstrip() for line in s.splitlines())
    s = s.strip()
    return s if s else None


def create_schema(con: sqlite3.Connection, with_fts: bool = True) -> None:
    cur = con.cursor()
    cur.executescript(SCHEMA_SQL)
    if with_fts:
        cur.executescript(FTS_SQL)
    con.commit()


def upsert_model_ids(cur: sqlite3.Cursor, model_names: Iterable[str]) -> Dict[str, int]:
    model_map: Dict[str, int] = {}
    for name in sorted(set(model_names)):
        if not name:
            continue
        cur.execute("INSERT OR IGNORE INTO models(name) VALUES (?)", (name,))
    cur.execute("SELECT id, name FROM models")
    for mid, name in cur.fetchall():
        model_map[name] = mid
    return model_map


def upsert_questions(cur: sqlite3.Cursor, questions: Iterable[str]) -> Dict[str, int]:
    qid_map: Dict[str, int] = {}
    now = dt.datetime.utcnow()
    for q in questions:
        cur.execute(
            "INSERT OR IGNORE INTO questions(text, created_at) VALUES (?, ?)",
            (q, now),
        )
    cur.execute("SELECT id, text FROM questions")
    for qid, text in cur.fetchall():
        qid_map[text] = qid
    return qid_map


def insert_or_update_answers(
    cur: sqlite3.Cursor,
    rows: Iterable[Dict[str, str]],
    qid_map: Dict[str, int],
    model_map: Dict[str, int],
) -> int:
    """Insert or update answers; returns affected row count."""
    now = dt.datetime.utcnow()
    n = 0
    for r in rows:
        q = r["question_text"]
        m = r["model_name"]
        a = r["answer_text"]
        if not (q and m and a):
            continue
        cur.execute(
            """
            INSERT INTO answers(question_id, model_id, answer_text, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(question_id, model_id)
            DO UPDATE SET answer_text=excluded.answer_text,
                          created_at=excluded.created_at
            """,
            (qid_map[q], model_map[m], a, now),
        )
        n += 1
    return n


def read_table(
    path: Path,
    sheet: Optional[str],
    question_col: Optional[str],
    keep_empty_answers: bool = False,
) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        # For .xlsx, requires openpyxl; for .xls, xlrd<=1.2.0
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    elif path.suffix.lower() == ".csv":
        # Try to sniff delimiter
        with path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
            sniffer = csv.Sniffer()
            sample = f.read(4096)
            f.seek(0)
            delimiter = sniffer.sniff(sample).delimiter if sniffer.has_header(sample) else ","
        df = pd.read_csv(path, delimiter=delimiter, encoding="utf-8", engine="python")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Ensure we have at least two columns
    if len(df.columns) < 2:
        raise ValueError("Expected at least 2 columns: [question, model1, model2, ...]")

    # Determine question column
    if question_col and question_col in df.columns:
        qcol = question_col
    else:
        qcol = df.columns[0]  # default: first column
    df = df.rename(columns={qcol: "question_text"})

    # Drop rows with no question text
    df["question_text"] = df["question_text"].apply(clean_text)
    df = df.dropna(subset=["question_text"])

    # Model columns: everything after question_text
    model_cols = [c for c in df.columns if c != "question_text"]

    # Melt wide -> long
    long_df = df.melt(
        id_vars=["question_text"],
        value_vars=model_cols,
        var_name="model_name",
        value_name="answer_text",
    )

    # Clean
    long_df["model_name"] = long_df["model_name"].astype(str).str.strip()
    long_df["answer_text"] = long_df["answer_text"].apply(clean_text)

    if not keep_empty_answers:
        long_df = long_df.dropna(subset=["answer_text"])

    return long_df.reset_index(drop=True)


# --------------------------- CLI entry --------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Load multi-LLM Q&A spreadsheet into a SQLite database."
    )
    ap.add_argument("--infile", required=True, help="Path to .xlsx/.xls/.csv")
    ap.add_argument("--sheet", help="Excel sheet name or index (0-based)")
    ap.add_argument("--db", default="llm_qna.db", help="SQLite DB path")
    ap.add_argument("--question-col", help="Explicit name of the question column")
    ap.add_argument("--no-fts", action="store_true", help="Disable FTS on questions")
    ap.add_argument("--dry-run", action="store_true", help="Parse only; no DB writes")
    args = ap.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        raise SystemExit(f"File not found: {infile}")

    # Parse sheet arg
    sheet: Optional[str]
    if args.sheet is None:
        sheet = None
    else:
        # accept index or name
        if args.sheet.isdigit():
            sheet = int(args.sheet)
        else:
            sheet = args.sheet

    print(f"Reading: {infile}")
    if sheet is not None:
        print(f"  Sheet: {sheet}")

    long_df = read_table(infile, sheet, args.question_col)
    print(f"Parsed rows (question–model pairs): {len(long_df)}")

    # Preview a few rows in dry-run
    if args.dry_run:
        print(long_df.head(8).to_string(index=False))
        print("Dry run complete. No database changes made.")
        return

    db_path = Path(args.db)
    con = sqlite3.connect(db_path)
    try:
        create_schema(con, with_fts=not args.no_fts)
        cur = con.cursor()

        # Upsert models
        model_names = long_df["model_name"].dropna().unique().tolist()
        model_map = upsert_model_ids(cur, model_names)

        # Upsert questions
        questions = long_df["question_text"].dropna().unique().tolist()
        qid_map = upsert_questions(cur, questions)

        # Insert/update answers
        rows = long_df.to_dict(orient="records")
        n = insert_or_update_answers(cur, rows, qid_map, model_map)

        con.commit()
        print(f"Loaded into: {db_path}")
        print(f"  Questions: {len(qid_map)}")
        print(f"  Models:    {len(model_map)}")
        print(f"  Answers:   {n}")

        # Quick verification query
        q_count = cur.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        a_count = cur.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
        print(f"DB counts → questions={q_count}, answers={a_count}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
