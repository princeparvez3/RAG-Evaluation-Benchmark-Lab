
import argparse
import json
from pathlib import Path

from app.core.dataset_loader import load_benchmark_dataset
from app.core.knowledge_base_loader import load_knowledge_base
from app.core.retriever import TfidfRetriever
from app.services.retrieval_evaluator import evaluate_retrieval


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_benchmark(
    k: int = 3,
    output_path: str | Path | None = None,
) -> tuple[dict, Path]:
    """Run retrieval evaluation and write a JSON report."""

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    dataset_path = PROJECT_ROOT / "data" / "samples" / "benchmark.jsonl"
    knowledge_path = (
        PROJECT_ROOT / "data" / "knowledge_base" / "documents.jsonl"
    )

    samples = load_benchmark_dataset(dataset_path)
    documents = load_knowledge_base(knowledge_path)

    retriever = TfidfRetriever(documents)
    report = evaluate_retrieval(samples, retriever, k=k)

    report["retriever"] = "TF-IDF"
    report["evaluation_type"] = "retrieval_only"

    if output_path is None:
        report_path = (
            PROJECT_ROOT
            / "reports"
            / "retrieval_benchmark_report.json"
        )
    else:
        report_path = Path(output_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    return report, report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the RAG retrieval benchmark."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top retrieval results to evaluate (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for the JSON report.",
    )

    args = parser.parse_args()

    try:
        report, report_path = run_benchmark(
            k=args.k,
            output_path=args.output,
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    metrics = report["aggregate_metrics"]

    print("\n=== RAG Retrieval Benchmark ===")
    print(f"Retriever: {report['retriever']}")
    print(f"Evaluation type: {report['evaluation_type']}")
    print(f"Questions evaluated: {report['total_samples']}")
    print(f"K: {report['k']}")
    print(f"Precision@{report['k']}: {metrics['precision_at_k']:.4f}")
    print(f"Recall@{report['k']}: {metrics['recall_at_k']:.4f}")
    print(f"MRR@{report['k']}: {metrics['mrr_at_k']:.4f}")

    print("\n=== Per-question results ===")

    for result in report["results"]:
        retrieved_ids = ", ".join(result["retrieved_source_ids"]) or "None"

        print(
            f"{result['sample_id']} | "
            f"Category: {result['category']} | "
            f"Precision: {result['precision_at_k']:.4f} | "
            f"Recall: {result['recall_at_k']:.4f} | "
            f"RR: {result['reciprocal_rank_at_k']:.4f}"
        )
        print(f"  Retrieved sources: {retrieved_ids}")

    print(f"\nJSON report saved to: {report_path}")


if __name__ == "__main__":
    main()
