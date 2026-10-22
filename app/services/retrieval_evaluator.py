
from statistics import mean
from typing import Protocol, Sequence

from app.metrics.retrieval_metrics import calculate_retrieval_metrics
from app.models.benchmark import BenchmarkSample
from app.models.document import KnowledgeDocument


class RetrievalItem(Protocol):
    document: KnowledgeDocument
    score: float


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> Sequence[RetrievalItem]:
        ...


def evaluate_retrieval(
    samples: list[BenchmarkSample],
    retriever: Retriever,
    k: int = 3,
) -> dict:
    """Evaluate a retrieval algorithm across benchmark samples."""

    if not samples:
        raise ValueError("At least one benchmark sample is required.")

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    results = []

    for sample in samples:
        expected_source_ids = sample.expected_citations

        if not expected_source_ids:
            raise ValueError(
                f"Sample '{sample.sample_id}' has no expected citations."
            )

        retrieved = retriever.retrieve(sample.question, top_k=k)

        retrieved_source_ids = [
            item.document.source_id for item in retrieved
        ]

        metrics = calculate_retrieval_metrics(
            expected_source_ids=expected_source_ids,
            retrieved_source_ids=retrieved_source_ids,
            k=k,
        )

        results.append(
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "category": sample.category,
                "expected_source_ids": expected_source_ids,
                "retrieved_source_ids": retrieved_source_ids,
                "retrieved_documents": [
                    {
                        "source_id": item.document.source_id,
                        "title": item.document.title,
                        "score": item.score,
                    }
                    for item in retrieved
                ],
                **metrics,
            }
        )

    aggregate_metrics = {
        "precision_at_k": mean(
            item["precision_at_k"] for item in results
        ),
        "recall_at_k": mean(
            item["recall_at_k"] for item in results
        ),
        "mrr_at_k": mean(
            item["reciprocal_rank_at_k"] for item in results
        ),
    }

    return {
        "total_samples": len(results),
        "k": k,
        "aggregate_metrics": aggregate_metrics,
        "results": results,
    }
