import argparse
import csv
import sqlite3
from pathlib import Path


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]

def _find_db_candidates() -> list[Path]:
    candidates: list[Path] = []
    for base in (Path("data"), Path(".")):
        if not base.exists():
            continue
        candidates.extend(sorted(p for p in base.rglob("*.db") if p.is_file()))
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def export_table(db_path: Path, csv_path: Path, table: str) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(f'SELECT * FROM "{table}"')
        column_names = [d[0] for d in cursor.description]

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(column_names)
            writer.writerows(cursor.fetchall())
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a SQLite table to CSV (defaults are tailored for this workshop repo)."
    )
    parser.add_argument(
        "--db",
        default="data/book_details.db",
        help="Path to the SQLite .db file (default: data/book_details.db).",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Table name to export. If omitted and the DB has exactly 1 table, that table is exported.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path. If omitted, uses data/<db_stem>.csv.",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        candidates = _find_db_candidates()
        hint = ""
        if candidates:
            shown = "\n".join(f"  - {p.as_posix()}" for p in candidates[:10])
            suffix = "\n  - ..." if len(candidates) > 10 else ""
            hint = (
                "\n\nAvailable .db files found in this repo:\n"
                f"{shown}{suffix}\n\nTry:\n"
                f"  python scripts/export_sqlite_to_csv.py --db {candidates[0].as_posix()}"
            )
        raise SystemExit(f"DB not found: {db_path}{hint}")

    out_path = Path(args.out) if args.out else Path("data") / f"{db_path.stem}.csv"

    conn = sqlite3.connect(str(db_path))
    try:
        tables = _list_tables(conn)
    finally:
        conn.close()

    if not tables:
        raise SystemExit(f"No tables found in {db_path}")

    table = args.table
    if table is None:
        if len(tables) != 1:
            raise SystemExit(
                f"DB has multiple tables ({', '.join(tables)}). Re-run with --table <name>."
            )
        table = tables[0]
    elif table not in tables:
        raise SystemExit(f"Table not found: {table}. Available: {', '.join(tables)}")

    export_table(db_path, out_path, table)
    print(f"Exported {db_path} ({table}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
