
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.services.failure_analyzer import analyze_retrieval_failures


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "evaluation_runs.sqlite3"

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
USE_POSTGRES = bool(DATABASE_URL)

_psycopg = None
_dict_row = None

if USE_POSTGRES:
    try:
        import psycopg
        from psycopg.rows import dict_row

        _psycopg = psycopg
        _dict_row = dict_row

    except ImportError as exc:
        raise RuntimeError(
            "DATABASE_URL is configured, but psycopg is not installed. "
            "Install it with: pip install 'psycopg[binary]'"
        ) from exc


if USE_POSTGRES:
    DATABASE_ERRORS = (sqlite3.Error, _psycopg.Error)
else:
    DATABASE_ERRORS = (sqlite3.Error,)


def _connect():
    """Connect to PostgreSQL when configured; otherwise use SQLite."""

    if USE_POSTGRES:
        return _psycopg.connect(
            DATABASE_URL,
            connect_timeout=10,
            row_factory=_dict_row,
        )

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _managed_connection() -> Iterator[Any]:
    """Commit successful operations and close the connection safely."""

    connection = _connect()

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def _placeholder() -> str:
    """Return the correct SQL parameter placeholder."""

    return "%s" if USE_POSTGRES else "?"


def initialize_database() -> None:
    """Create the evaluation history table if it does not exist."""

    with _managed_connection() as connection:
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


def _summary_from_row(row: Any) -> dict[str, Any]:
    """Convert a stored database record into a summary."""

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
    """Save a completed benchmark and its failure analysis."""

    required_fields = (
        "k",
        "dataset",
        "knowledge_base",
        "total_samples",
        "knowledge_base_document_count",
        "retrievers",
    )

    if not isinstance(report, dict) or any(
        field not in report for field in required_fields
    ):
        raise ValueError(
            "The benchmark report is missing required fields."
        )

    # Use the already imported function directly.
    failure_analysis = analyze_retrieval_failures(report)

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    )

    report_json = json.dumps(report, ensure_ascii=False)
    failure_json = json.dumps(failure_analysis, ensure_ascii=False)

    marker = _placeholder()

    initialize_database()

    with _managed_connection() as connection:
        connection.execute(
            f"""
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
            VALUES (
                {marker}, {marker}, {marker}, {marker}, {marker},
                {marker}, {marker}, {marker}, {marker}
            )
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
            f"""
            SELECT *
            FROM evaluation_runs
            WHERE run_id = {marker}
            """,
            (run_id,),
        ).fetchone()

    if row is None:
        raise RuntimeError(
            "The saved evaluation run could not be retrieved."
        )

    return _summary_from_row(row)


def list_evaluation_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Return saved evaluation runs, newest first."""

    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer.")

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")

    initialize_database()

    marker = _placeholder()

    with _managed_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT *
            FROM evaluation_runs
            ORDER BY created_at DESC, run_id DESC
            LIMIT {marker}
            """,
            (limit,),
        ).fetchall()

    return [_summary_from_row(row) for row in rows]


def get_evaluation_run(run_id: str) -> dict[str, Any] | None:
    """Retrieve one run with its report and failure analysis."""

    initialize_database()

    marker = _placeholder()

    with _managed_connection() as connection:
        row = connection.execute(
            f"""
            SELECT *
            FROM evaluation_runs
            WHERE run_id = {marker}
            """,
            (run_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        **_summary_from_row(row),
        "report": json.loads(row["report_json"]),
        "failure_analysis": json.loads(
            row["failure_analysis_json"]
        ),
    }

def get_latest_evaluation_run() -> dict[str, Any] | None:
    """Retrieve the latest saved run, including its full report."""

    initialize_database()
    marker = _placeholder()

    with _managed_connection() as connection:
        row = connection.execute(
            f"""
            SELECT run_id
            FROM evaluation_runs
            ORDER BY created_at DESC, run_id DESC
            LIMIT {marker}
            """,
            (1,),
        ).fetchone()

    if row is None:
        return None

    return get_evaluation_run(row["run_id"])
