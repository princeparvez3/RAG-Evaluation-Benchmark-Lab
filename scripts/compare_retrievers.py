
import argparse
import json
from pathlib import Path

from app.core.bm25_retriever import BM25Retriever
from app.core.dataset_loader import load_benchmark_dataset
from app.core.knowledge_base_loader import load_knowledge_base
from app.core.retriever import TfidfRetriever
from app.services.retrieval_evaluator import evaluate_retrieval


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "samples" / "benchmark.jsonl"
)

DEFAULT_KNOWLEDGE_BASE = (
    PROJECT_ROOT / "data" / "knowledge_base" / "documents.jsonl"
)


def run_comparison(
    k: int = 3,
    output_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    knowledge_base_path: str | Path | None = None,
) -> tuple[dict, Path]:
    """Compare TF-IDF and BM25 on the same dataset and knowledge base."""

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    dataset_file = (
        Path(dataset_path)
        if dataset_path is not None
        else DEFAULT_DATASET
    )

    knowledge_file = (
        Path(knowledge_base_path)
        if knowledge_base_path is not None
        else DEFAULT_KNOWLEDGE_BASE
    )

    samples = load_benchmark_dataset(dataset_file)
    documents = load_knowledge_base(knowledge_file)

    retrievers = {
        "TF-IDF": TfidfRetriever(documents),
        "BM25": BM25Retriever(documents),
    }

    comparison = {}

    for name, retriever in retrievers.items():
        result = evaluate_retrieval(samples, retriever, k=k)

        comparison[name] = {
            "aggregate_metrics": result["aggregate_metrics"],
            "per_question": result["results"],
        }

    report = {
        "evaluation_type": "retrieval_only",
        "dataset": dataset_file.stem,
        "knowledge_base": knowledge_file.stem,
        "total_samples": len(samples),
        "knowledge_base_document_count": len(documents),
        "k": k,
        "retrievers": comparison,
    }

    if output_path is None:
        report_path = (
            PROJECT_ROOT
            / "reports"
            / (
                f"{dataset_file.stem}_"
                f"{knowledge_file.stem}_"
                "retriever_comparison.json"
            )
        )
    else:
        report_path = Path(output_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    return report, report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF and BM25 retrieval performance."
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top results to evaluate (default: 3).",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the benchmark JSONL file.",
    )

    parser.add_argument(
        "--knowledge-base",
        type=str,
        default=None,
        help="Path to the knowledge-base JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON report output path.",
    )

    args = parser.parse_args()

    try:
        report, report_path = run_comparison(
            k=args.k,
            output_path=args.output,
            dataset_path=args.dataset,
            knowledge_base_path=args.knowledge_base,
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print("\n=== Retrieval Algorithm Comparison ===")
    print(f"Dataset: {report['dataset']}")
    print(f"Knowledge base: {report['knowledge_base']}")
    print(f"Dataset questions: {report['total_samples']}")
    print(
        "Knowledge-base documents: "
        f"{report['knowledge_base_document_count']}"
    )
    print(f"K: {report['k']}\n")

    print(
        f"{'Retriever':<12}"
        f"{'Precision@K':>15}"
        f"{'Recall@K':>13}"
        f"{'MRR@K':>12}"
    )

    for name, result in report["retrievers"].items():
        metrics = result["aggregate_metrics"]

        print(
            f"{name:<12}"
            f"{metrics['precision_at_k']:>15.4f}"
            f"{metrics['recall_at_k']:>13.4f}"
            f"{metrics['mrr_at_k']:>12.4f}"
        )

    print("\n=== Per-question top results ===")

    for name, result in report["retrievers"].items():
        print(f"\n{name}")

        for item in result["per_question"]:
            top_source = (
                item["retrieved_source_ids"][0]
                if item["retrieved_source_ids"]
                else "None"
            )

            print(
                f"{item['sample_id']}: "
                f"top source={top_source}, "
                f"RR={item['reciprocal_rank_at_k']:.4f}"
            )

    print(f"\nComparison report saved to: {report_path}")


if __name__ == "__main__":
    main()
