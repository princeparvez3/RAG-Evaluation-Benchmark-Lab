
import pytest

from app.metrics.retrieval_metrics import (
    calculate_retrieval_metrics,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_precision_at_k():
    score = precision_at_k(
        expected_source_ids=["doc-A", "doc-C"],
        retrieved_source_ids=["doc-A", "doc-X", "doc-C"],
        k=2,
    )

    assert score == pytest.approx(0.5)


def test_precision_uses_fixed_k_denominator():
    score = precision_at_k(
        expected_source_ids=["doc-A"],
        retrieved_source_ids=["doc-A"],
        k=3,
    )

    assert score == pytest.approx(1 / 3)


def test_recall_at_k():
    score = recall_at_k(
        expected_source_ids=["doc-A", "doc-C"],
        retrieved_source_ids=["doc-A", "doc-X"],
        k=2,
    )

    assert score == pytest.approx(0.5)


def test_reciprocal_rank_at_k():
    score = reciprocal_rank_at_k(
        expected_source_ids=["doc-A", "doc-C"],
        retrieved_source_ids=["doc-X", "doc-C", "doc-A"],
        k=2,
    )

    assert score == pytest.approx(0.5)


def test_reciprocal_rank_is_zero_when_no_relevant_result():
    score = reciprocal_rank_at_k(
        expected_source_ids=["doc-A"],
        retrieved_source_ids=["doc-X", "doc-Y"],
        k=2,
    )

    assert score == 0.0


def test_empty_expected_sources_are_rejected():
    with pytest.raises(ValueError, match="At least one expected"):
        precision_at_k(
            expected_source_ids=[],
            retrieved_source_ids=["doc-A"],
            k=1,
        )


def test_invalid_k_is_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k(
            expected_source_ids=["doc-A"],
            retrieved_source_ids=["doc-A"],
            k=0,
        )


def test_combined_metrics():
    metrics = calculate_retrieval_metrics(
        expected_source_ids=["doc-A", "doc-C"],
        retrieved_source_ids=["doc-X", "doc-A", "doc-C"],
        k=2,
    )

    assert metrics["precision_at_k"] == pytest.approx(0.5)
    assert metrics["recall_at_k"] == pytest.approx(0.5)
    assert metrics["reciprocal_rank_at_k"] == pytest.approx(0.5)
