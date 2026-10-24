
import json
import sqlite3
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.services.failure_analyzer import analyze_retrieval_failures
from app.services.run_history import (
    get_evaluation_run,
    list_evaluation_runs,
    save_evaluation_run,
)
from scripts.compare_retrievers import run_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT / "reports" / "api_retriever_comparison.json"
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
        "failure analysis, JSON reports, and persistent run history."
    ),
    version="1.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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
        "version": "1.4.0",
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
    """Run a selected benchmark and save its report and history."""

    config = DATASETS[dataset]

    try:
        report, _ = run_comparison(
            k=k,
            output_path=REPORT_PATH,
            dataset_path=config["dataset_path"],
            knowledge_base_path=config["knowledge_base_path"],
        )

        saved_run = save_evaluation_run(report)

    except (ValueError, OSError, sqlite3.Error, RuntimeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark execution or history save failed: {exc}",
        ) from exc

    return {
        "status": "completed",
        "report_file": "reports/api_retriever_comparison.json",
        "run": saved_run,
        "report": report,
    }


@app.get("/api/v1/benchmark/latest", tags=["Benchmark"])
def get_latest_benchmark():
    """Return the latest benchmark JSON report."""

    if not REPORT_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="No benchmark report exists yet. Run a benchmark first.",
        )

    try:
        with REPORT_PATH.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read the benchmark report: {exc}",
        ) from exc

    return {
        "status": "completed",
        "report_file": "reports/api_retriever_comparison.json",
        "report": report,
    }


@app.get("/api/v1/benchmark/failures", tags=["Failure Analysis"])
def get_failure_analysis():
    """Analyze retrieval outcomes from the latest benchmark."""

    if not REPORT_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="No benchmark report exists yet. Run a benchmark first.",
        )

    try:
        with REPORT_PATH.open("r", encoding="utf-8") as file:
            report = json.load(file)

        analysis = analyze_retrieval_failures(report)

    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failure analysis could not be completed: {exc}",
        ) from exc

    return {
        "status": "completed",
        "source_report": REPORT_PATH.name,
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
    except (ValueError, OSError, sqlite3.Error, json.JSONDecodeError) as exc:
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
    except (ValueError, OSError, sqlite3.Error, json.JSONDecodeError) as exc:
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
