
import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.failure_analyzer import analyze_retrieval_failures


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "evaluation_runs.sqlite3"


def initialize_database() -> None:
    """Create the local SQLite database and history table if needed."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(DB_PATH, timeout=10)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    k INTEGER NOT NULL,
                    dataset TEXT NOT NULL,
                    knowledge_base TEXT NOT NULL,
                    total_samples INTEGER NOT NULL,
                    knowledge_base_document_count INTEGER NOT NULL,
                    report_json TEXT NOT NULL,
                    failure_analysis_json TEXT NOT NULL
                )
                """
            )


def _summary_from_row(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a stored record into a compact history entry."""

    report = json.loads(row["report_json"])
    failure_analysis = json.loads(row["failure_analysis_json"])

    return {
        "run_id": row["run_id"],
        "created_at": row["created_at"],
        "k": row["k"],
        "dataset": row["dataset"],
        "knowledge_base": row["knowledge_base"],
        "total_samples": row["total_samples"],
        "knowledge_base_document_count": row[
            "knowledge_base_document_count"
        ],
        "retriever_metrics": {
            name: data["aggregate_metrics"]
            for name, data in report["retrievers"].items()
        },
        "failure_summary": {
            name: {
                "successes": data["successes"],
                "ranking_errors": data["ranking_errors"],
                "retrieval_misses": data["retrieval_misses"],
                "top1_success_rate": data["top1_success_rate"],
            }
            for name, data in failure_analysis.items()
        },
    }


def save_evaluation_run(report: dict[str, Any]) -> dict[str, Any]:
    """Persist a completed benchmark and its failure analysis."""

    required = (
        "k",
        "dataset",
        "knowledge_base",
        "total_samples",
        "knowledge_base_document_count",
        "retrievers",
    )

    if not isinstance(report, dict) or any(
        key not in report for key in required
    ):
        raise ValueError("The benchmark report is missing required fields.")

    failure_analysis = analyze_retrieval_failures(report)
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    report_json = json.dumps(report, ensure_ascii=False)
    failure_json = json.dumps(failure_analysis, ensure_ascii=False)

    initialize_database()

    with closing(sqlite3.connect(DB_PATH, timeout=10)) as connection:
        connection.row_factory = sqlite3.Row

        with connection:
            connection.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id,
                    created_at,
                    k,
                    dataset,
                    knowledge_base,
                    total_samples,
                    knowledge_base_document_count,
                    report_json,
                    failure_analysis_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    report["k"],
                    report["dataset"],
                    report["knowledge_base"],
                    report["total_samples"],
                    report["knowledge_base_document_count"],
                    report_json,
                    failure_json,
                ),
            )

            row = connection.execute(
                "SELECT * FROM evaluation_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()

    if row is None:
        raise RuntimeError("The saved evaluation run could not be retrieved.")

    return _summary_from_row(row)


def list_evaluation_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent benchmark runs, newest first."""

    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer.")

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")

    initialize_database()

    with closing(sqlite3.connect(DB_PATH, timeout=10)) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT *
            FROM evaluation_runs
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [_summary_from_row(row) for row in rows]


def get_evaluation_run(run_id: str) -> dict[str, Any] | None:
    """Retrieve one saved run, including full metrics and failure analysis."""

    initialize_database()

    with closing(sqlite3.connect(DB_PATH, timeout=10)) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            "SELECT * FROM evaluation_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        **_summary_from_row(row),
        "report": json.loads(row["report_json"]),
        "failure_analysis": json.loads(row["failure_analysis_json"]),
    }
