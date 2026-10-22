
def _validate_inputs(
    expected_source_ids: list[str],
    retrieved_source_ids: list[str],
    k: int,
) -> tuple[set[str], list[str]]:
    """Validate metric inputs and return normalized source IDs."""

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    if not isinstance(expected_source_ids, list):
        raise TypeError("expected_source_ids must be a list.")

    if not isinstance(retrieved_source_ids, list):
        raise TypeError("retrieved_source_ids must be a list.")

    all_ids = expected_source_ids + retrieved_source_ids

    if any(
        not isinstance(source_id, str) or not source_id.strip()
        for source_id in all_ids
    ):
        raise ValueError("Source IDs must be non-empty strings.")

    expected = {source_id.strip() for source_id in expected_source_ids}
    retrieved = [source_id.strip() for source_id in retrieved_source_ids]

    if not expected:
        raise ValueError(
            "At least one expected relevant source ID is required."
        )

    return expected, retrieved[:k]


def precision_at_k(
    expected_source_ids: list[str],
    retrieved_source_ids: list[str],
    k: int = 3,
) -> float:
    """Fraction of the top-k result slots containing relevant documents.

    The denominator is always k, even when fewer than k documents
    were retrieved.
    """

    expected, top_k = _validate_inputs(
        expected_source_ids, retrieved_source_ids, k
    )

    relevant_hits = len(set(top_k) & expected)
    return relevant_hits / k


def recall_at_k(
    expected_source_ids: list[str],
    retrieved_source_ids: list[str],
    k: int = 3,
) -> float:
    """Fraction of expected relevant documents found in the top-k results."""

    expected, top_k = _validate_inputs(
        expected_source_ids, retrieved_source_ids, k
    )

    relevant_hits = len(set(top_k) & expected)
    return relevant_hits / len(expected)


def reciprocal_rank_at_k(
    expected_source_ids: list[str],
    retrieved_source_ids: list[str],
    k: int = 3,
) -> float:
    """Return the reciprocal rank of the first relevant result.

    Returns 1/rank for the first relevant result within top k,
    or 0.0 if no relevant result is found.
    """

    expected, top_k = _validate_inputs(
        expected_source_ids, retrieved_source_ids, k
    )

    for rank, source_id in enumerate(top_k, start=1):
        if source_id in expected:
            return 1.0 / rank

    return 0.0


def calculate_retrieval_metrics(
    expected_source_ids: list[str],
    retrieved_source_ids: list[str],
    k: int = 3,
) -> dict[str, float]:
    """Calculate retrieval metrics for a single benchmark question."""

    return {
        "precision_at_k": precision_at_k(
            expected_source_ids, retrieved_source_ids, k
        ),
        "recall_at_k": recall_at_k(
            expected_source_ids, retrieved_source_ids, k
        ),
        "reciprocal_rank_at_k": reciprocal_rank_at_k(
            expected_source_ids, retrieved_source_ids, k
        ),
    }
