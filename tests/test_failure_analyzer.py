
import pytest

from app.services.failure_analyzer import analyze_retrieval_failures


def test_classifies_success_ranking_error_and_miss():
    report = {
        "retrievers": {
            "Example": {
                "per_question": [
                    {
                        "sample_id": "Q1",
                        "category": "security",
                        "expected_source_ids": ["doc-a"],
                        "retrieved_source_ids": ["doc-a", "doc-b"],
                    },
                    {
                        "sample_id": "Q2",
                        "category": "limits",
                        "expected_source_ids": ["doc-c"],
                        "retrieved_source_ids": ["doc-b", "doc-c"],
                    },
                    {
                        "sample_id": "Q3",
                        "category": "cost",
                        "expected_source_ids": ["doc-d"],
                        "retrieved_source_ids": ["doc-a", "doc-b"],
                    },
                ]
            }
        }
    }

    result = analyze_retrieval_failures(report)["Example"]

    assert result["successes"] == 1
    assert result["ranking_errors"] == 1
    assert result["retrieval_misses"] == 1
    assert result["top1_success_rate"] == pytest.approx(1 / 3)


def test_empty_retrieved_results_are_a_miss():
    report = {
        "retrievers": {
            "Example": {
                "per_question": [
                    {
                        "sample_id": "Q1",
                        "category": "cost",
                        "expected_source_ids": ["doc-a"],
                        "retrieved_source_ids": [],
                    }
                ]
            }
        }
    }

    result = analyze_retrieval_failures(report)["Example"]

    assert result["retrieval_misses"] == 1


def test_empty_report_is_rejected():
    with pytest.raises(ValueError, match="at least one retriever"):
        analyze_retrieval_failures({"retrievers": {}})


def test_missing_expected_sources_are_rejected():
    report = {
        "retrievers": {
            "Example": {
                "per_question": [
                    {
                        "sample_id": "Q1",
                        "category": "cost",
                        "expected_source_ids": [],
                        "retrieved_source_ids": ["doc-a"],
                    }
                ]
            }
        }
    }

    with pytest.raises(ValueError, match="no expected sources"):
        analyze_retrieval_failures(report)
