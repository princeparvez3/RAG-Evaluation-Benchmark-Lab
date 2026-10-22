
import json
from pathlib import Path

import pytest

from scripts.compare_retrievers import run_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_comparison_runs_both_retrievers(tmp_path):
    report, saved_path = run_comparison(
        k=3,
        output_path=tmp_path / "comparison.json",
    )

    assert saved_path.is_file()
    assert report["total_samples"] == 5
    assert report["dataset"] == "benchmark"
    assert set(report["retrievers"]) == {"TF-IDF", "BM25"}


def test_comparison_report_contains_all_metrics(tmp_path):
    report, _ = run_comparison(
        k=3,
        output_path=tmp_path / "metrics.json",
    )

    for result in report["retrievers"].values():
        metrics = result["aggregate_metrics"]

        assert "precision_at_k" in metrics
        assert "recall_at_k" in metrics
        assert "mrr_at_k" in metrics


def test_comparison_saves_per_question_results(tmp_path):
    output_path = tmp_path / "per_question.json"

    run_comparison(k=3, output_path=output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))

    for result in saved["retrievers"].values():
        assert len(result["per_question"]) == 5
        assert all(
            "retrieved_source_ids" in item
            for item in result["per_question"]
        )


def test_comparison_rejects_invalid_k(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        run_comparison(
            k=0,
            output_path=tmp_path / "invalid.json",
        )


def test_challenge_dataset_uses_all_15_samples(tmp_path):
    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "samples"
        / "benchmark_challenge.jsonl"
    )

    report, saved_path = run_comparison(
        k=3,
        output_path=tmp_path / "challenge.json",
        dataset_path=dataset_path,
    )

    assert report["dataset"] == "benchmark_challenge"
    assert report["total_samples"] == 15
    assert saved_path.is_file()

    for result in report["retrievers"].values():
        assert len(result["per_question"]) == 15


def test_challenge_report_contains_both_algorithms(tmp_path):
    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "samples"
        / "benchmark_challenge.jsonl"
    )
    output_path = tmp_path / "challenge.json"

    run_comparison(
        dataset_path=dataset_path,
        output_path=output_path,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert set(saved["retrievers"]) == {"TF-IDF", "BM25"}
    assert saved["total_samples"] == 15


def test_expanded_knowledge_base_comparison(tmp_path):
    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "samples"
        / "benchmark_challenge.jsonl"
    )

    knowledge_base_path = (
        PROJECT_ROOT
        / "data"
        / "knowledge_base"
        / "documents_expanded.jsonl"
    )

    output_path = tmp_path / "expanded_comparison.json"

    report, saved_path = run_comparison(
        k=3,
        output_path=output_path,
        dataset_path=dataset_path,
        knowledge_base_path=knowledge_base_path,
    )

    assert saved_path.is_file()
    assert report["total_samples"] == 15
    assert report["knowledge_base_document_count"] == 15
    assert report["knowledge_base"] == "documents_expanded"

    for result in report["retrievers"].values():
        assert len(result["per_question"]) == 15


def test_missing_knowledge_base_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        run_comparison(
            k=3,
            output_path=tmp_path / "missing.json",
            knowledge_base_path=tmp_path / "missing_documents.jsonl",
        )
