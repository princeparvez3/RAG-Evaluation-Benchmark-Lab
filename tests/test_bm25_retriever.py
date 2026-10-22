
from pathlib import Path

import pytest

from app.core.bm25_retriever import BM25Retriever
from app.core.knowledge_base_loader import load_knowledge_base


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def retriever():
    path = (
        PROJECT_ROOT
        / "data"
        / "knowledge_base"
        / "documents.jsonl"
    )
    documents = load_knowledge_base(path)
    return BM25Retriever(documents)


@pytest.mark.parametrize(
    ("question", "expected_source"),
    [
        (
            "Which file formats does the platform support for exporting datasets?",
            "doc-export-01",
        ),
        (
            "For how long are benchmark results retained by default?",
            "doc-retention-01",
        ),
        (
            "What is the maximum number of prompts in one evaluation batch?",
            "doc-limits-01",
        ),
        (
            "Where should API keys be stored?",
            "doc-security-01",
        ),
        (
            "How does the platform estimate model API costs?",
            "doc-cost-01",
        ),
    ],
)
def test_bm25_finds_expected_source(
    retriever,
    question,
    expected_source,
):
    results = retriever.retrieve(question, top_k=3)

    assert results
    assert results[0].document.source_id == expected_source


def test_results_are_sorted_by_descending_score(retriever):
    results = retriever.retrieve(
        "benchmark results evaluation cost",
        top_k=5,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results(retriever):
    results = retriever.retrieve(
        "evaluation platform",
        top_k=1,
    )

    assert len(results) <= 1


def test_empty_query_is_rejected(retriever):
    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.retrieve("   ")


def test_unknown_terms_return_no_results(retriever):
    results = retriever.retrieve("qzxq blorf xyzz")

    assert results == []


def test_empty_document_list_is_rejected():
    with pytest.raises(
        ValueError,
        match="At least one knowledge document",
    ):
        BM25Retriever([])


def test_non_word_query_returns_no_results(retriever):
    results = retriever.retrieve("?!...")

    assert results == []
