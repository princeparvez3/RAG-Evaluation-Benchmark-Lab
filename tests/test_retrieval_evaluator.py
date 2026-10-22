
from pathlib import Path

import pytest

from app.core.knowledge_base_loader import load_knowledge_base
from app.core.retriever import TfidfRetriever
from app.core.dataset_loader import load_benchmark_dataset
from app.models.benchmark import BenchmarkSample
from app.services.retrieval_evaluator import evaluate_retrieval


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evaluation_setup():
    samples = load_benchmark_dataset(
        PROJECT_ROOT / "data" / "samples" / "benchmark.jsonl"
    )

    documents = load_knowledge_base(
        PROJECT_ROOT
        / "data"
        / "knowledge_base"
        / "documents.jsonl"
    )

    retriever = TfidfRetriever(documents)

    return samples, retriever


def test_evaluate_project_benchmark(evaluation_setup):
    samples, retriever = evaluation_setup

    report = evaluate_retrieval(samples, retriever, k=3)

    assert report["total_samples"] == 5
    assert report["k"] == 3
    assert len(report["results"]) == 5

    for result in report["results"]:
        assert result["retrieved_source_ids"]
        assert (
            result["retrieved_source_ids"][0]
            in result["expected_source_ids"]
        )


def test_aggregate_metrics_are_correct(evaluation_setup):
    samples, retriever = evaluation_setup

    report = evaluate_retrieval(samples, retriever, k=3)
    metrics = report["aggregate_metrics"]

    assert metrics["precision_at_k"] == pytest.approx(1 / 3)
    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["mrr_at_k"] == pytest.approx(1.0)


def test_results_include_document_scores(evaluation_setup):
    samples, retriever = evaluation_setup

    report = evaluate_retrieval(samples, retriever, k=3)

    for result in report["results"]:
        assert result["retrieved_documents"]

        for document in result["retrieved_documents"]:
            assert document["source_id"]
            assert document["title"]
            assert 0.0 <= document["score"] <= 1.0


def test_empty_samples_are_rejected(evaluation_setup):
    _, retriever = evaluation_setup

    with pytest.raises(ValueError, match="At least one"):
        evaluate_retrieval([], retriever)


def test_missing_expected_citations_are_rejected(evaluation_setup):
    _, retriever = evaluation_setup

    sample = BenchmarkSample(
        sample_id="RAG-NO-CITATION",
        question="What is a benchmark?",
        reference_answer="A benchmark is used for comparison.",
    )

    with pytest.raises(ValueError, match="no expected citations"):
        evaluate_retrieval([sample], retriever)


def test_invalid_k_is_rejected(evaluation_setup):
    samples, retriever = evaluation_setup

    with pytest.raises(ValueError, match="positive integer"):
        evaluate_retrieval(samples, retriever, k=0)
