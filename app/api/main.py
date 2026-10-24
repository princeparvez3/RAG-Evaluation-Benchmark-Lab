
import json
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.services.failure_analyzer import analyze_retrieval_failures
from app.services.run_history import (
    DATABASE_ERRORS,
    get_evaluation_run,
    get_latest_evaluation_run,
    list_evaluation_runs,
    save_evaluation_run,
)
from scripts.compare_retrievers import run_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT / "reports" / "api_retriever_comparison.json"
)

# Local development saves reports to disk by default.
# Vercel disables report-file writes unless explicitly configured.
_default_persist = "false" if os.getenv("VERCEL") else "true"

PERSIST_REPORT_FILES = (
    os.getenv("PERSIST_REPORT_FILES", _default_persist)
    .strip()
    .lower()
    in {"1", "true", "yes"}
)

REPORT_LOCATION = (
    "reports/api_retriever_comparison.json"
    if PERSIST_REPORT_FILES
    else "Saved in evaluation history database"
)

DATASETS = {
    "default": {
        "dataset_path": (
            PROJECT_ROOT / "data" / "samples" / "benchmark.jsonl"
        ),
        "knowledge_base_path": (
            PROJECT_ROOT
            / "data"
            / "knowledge_base"
            / "documents.jsonl"
        ),
    },
    "expanded": {
        "dataset_path": (
            PROJECT_ROOT
            / "data"
            / "samples"
            / "benchmark_expanded.jsonl"
        ),
        "knowledge_base_path": (
            PROJECT_ROOT
            / "data"
            / "knowledge_base"
            / "documents_expanded.jsonl"
        ),
    },
}

app = FastAPI(
    title="RAG Evaluation & Benchmark Lab API",
    description=(
        "Retrieval benchmarking, TF-IDF vs BM25 comparison, "
        "failure analysis, JSON reports, and evaluation history."
    ),
    version="1.5.0",
)

allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip().rstrip("/")

if frontend_origin and frontend_origin not in allowed_origins:
    allowed_origins.append(frontend_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

API_ERRORS = DATABASE_ERRORS + (
    ValueError,
    OSError,
    RuntimeError,
    json.JSONDecodeError,
)


@app.get("/", tags=["Health"])
def root():
    return {
        "message": "RAG Evaluation & Benchmark Lab API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}


@app.get("/api/v1/info", tags=["Info"])
def project_info():
    return {
        "project": "RAG Evaluation & Benchmark Lab",
        "version": "1.5.0",
        "retrievers": ["TF-IDF", "BM25"],
        "metrics": [
            "Precision@K",
            "Recall@K",
            "MRR@K",
        ],
        "available_datasets": {
            "default": "5 questions and 5 documents",
            "expanded": "20 questions and 30 documents",
        },
        "benchmark_endpoint": "/api/v1/benchmark/run",
        "latest_report_endpoint": "/api/v1/benchmark/latest",
        "failure_analysis_endpoint": "/api/v1/benchmark/failures",
        "run_history_endpoint": "/api/v1/runs",
        "report_storage": (
            "database_and_local_file"
            if PERSIST_REPORT_FILES
            else "evaluation_history_database"
        ),
    }


@app.post("/api/v1/benchmark/run", tags=["Benchmark"])
def execute_benchmark(
    k: int = Query(
        default=3,
        ge=1,
        le=15,
        description="Number of top retrieval results to evaluate.",
    ),
    dataset: Literal["default", "expanded"] = Query(
        default="default",
        description="Choose the default or expanded benchmark dataset.",
    ),
):
    """Run a benchmark and save its result in evaluation history."""

    config = DATASETS[dataset]

    try:
        report, _ = run_comparison(
            k=k,
            output_path=REPORT_PATH,
            dataset_path=config["dataset_path"],
            knowledge_base_path=config["knowledge_base_path"],
            persist_report=PERSIST_REPORT_FILES,
        )

        saved_run = save_evaluation_run(report)

    except API_ERRORS as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark execution or history save failed: {exc}",
        ) from exc

    return {
        "status": "completed",
        "report_file": REPORT_LOCATION,
        "run": saved_run,
        "report": report,
    }


@app.get("/api/v1/benchmark/latest", tags=["Benchmark"])
def get_latest_benchmark():
    """Return the latest saved benchmark report."""

    try:
        latest_run = get_latest_evaluation_run()

        if latest_run is not None:
            return {
                "status": "completed",
                "report_file": REPORT_LOCATION,
                "run_id": latest_run["run_id"],
                "report": latest_run["report"],
            }

        # Backward-compatible fallback for older local reports.
        if PERSIST_REPORT_FILES and REPORT_PATH.is_file():
            with REPORT_PATH.open("r", encoding="utf-8") as file:
                report = json.load(file)

            return {
                "status": "completed",
                "report_file": REPORT_LOCATION,
                "report": report,
            }

    except API_ERRORS as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load the latest benchmark: {exc}",
        ) from exc

    raise HTTPException(
        status_code=404,
        detail="No saved benchmark exists yet. Run a benchmark first.",
    )


@app.get("/api/v1/benchmark/failures", tags=["Failure Analysis"])
def get_failure_analysis():
    """Return failure analysis for the latest saved benchmark."""

    try:
        latest_run = get_latest_evaluation_run()

        if latest_run is not None:
            analysis = latest_run["failure_analysis"]
            source_report = latest_run["dataset"]

        elif PERSIST_REPORT_FILES and REPORT_PATH.is_file():
            # Fallback for reports generated before history was added.
            with REPORT_PATH.open("r", encoding="utf-8") as file:
                report = json.load(file)

            analysis = analyze_retrieval_failures(report)
            source_report = REPORT_PATH.name

        else:
            raise HTTPException(
                status_code=404,
                detail="No saved benchmark exists yet. Run a benchmark first.",
            )

    except API_ERRORS as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failure analysis could not be loaded: {exc}",
        ) from exc

    return {
        "status": "completed",
        "source_report": source_report,
        "retrievers": analysis,
    }


@app.get("/api/v1/runs", tags=["Evaluation History"])
def get_run_history(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of recent runs to return.",
    ),
):
    """List saved evaluation runs, newest first."""

    try:
        runs = list_evaluation_runs(limit=limit)

    except API_ERRORS as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load evaluation history: {exc}",
        ) from exc

    return {
        "status": "success",
        "count": len(runs),
        "runs": runs,
    }


@app.get("/api/v1/runs/{run_id}", tags=["Evaluation History"])
def get_run_details(run_id: str):
    """Retrieve a saved run's full report and failure analysis."""

    try:
        run = get_evaluation_run(run_id)

    except API_ERRORS as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load evaluation run: {exc}",
        ) from exc

    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation run '{run_id}' was not found.",
        )

    return {
        "status": "success",
        "run": run,
    }
