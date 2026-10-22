
from collections import Counter


def analyze_retrieval_failures(comparison_report: dict) -> dict:
    """Classify retrieval outcomes and summarize failures by category."""

    retrievers = comparison_report.get("retrievers")

    if not isinstance(retrievers, dict) or not retrievers:
        raise ValueError("Report must contain at least one retriever.")

    analysis = {}

    for name, retriever_data in retrievers.items():
        questions = retriever_data.get("per_question")

        if not isinstance(questions, list) or not questions:
            raise ValueError(
                f"Retriever '{name}' must contain per-question results."
            )

        outcomes = []
        category_counts = Counter()

        for item in questions:
            expected_ids = item.get("expected_source_ids", [])
            retrieved_ids = item.get("retrieved_source_ids", [])
            category = item.get("category", "uncategorized")

            if not expected_ids:
                raise ValueError(
                    f"Sample '{item.get('sample_id')}' has no expected sources."
                )

            if not isinstance(retrieved_ids, list):
                raise ValueError("Retrieved source IDs must be a list.")

            if retrieved_ids and retrieved_ids[0] in expected_ids:
                status = "SUCCESS"
            elif any(source in expected_ids for source in retrieved_ids):
                status = "RANKING_ERROR"
            else:
                status = "RETRIEVAL_MISS"

            category_counts[(category, status)] += 1

            outcomes.append(
                {
                    "sample_id": item.get("sample_id"),
                    "category": category,
                    "expected_source_ids": expected_ids,
                    "retrieved_source_ids": retrieved_ids,
                    "status": status,
                }
            )

        total = len(outcomes)
        successes = sum(
            item["status"] == "SUCCESS" for item in outcomes
        )
        ranking_errors = sum(
            item["status"] == "RANKING_ERROR" for item in outcomes
        )
        misses = sum(
            item["status"] == "RETRIEVAL_MISS" for item in outcomes
        )

        analysis[name] = {
            "total_questions": total,
            "successes": successes,
            "ranking_errors": ranking_errors,
            "retrieval_misses": misses,
            "top1_success_rate": successes / total,
            "failures": [
                item for item in outcomes
                if item["status"] != "SUCCESS"
            ],
            "category_summary": [
                {
                    "category": category,
                    "status": status,
                    "count": count,
                }
                for (category, status), count
                in sorted(category_counts.items())
            ],
        }

    return analysis
