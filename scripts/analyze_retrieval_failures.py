import argparse
import json
from pathlib import Path

from app.services.failure_analyzer import analyze_retrieval_failures


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RAG retrieval benchmark failures."
    )

    parser.add_argument(
        "--report",
        default=(
            "reports/"
            "benchmark_challenge_documents_expanded_retriever_comparison.json"
        ),
        help="Path to the retriever comparison JSON report.",
    )

    parser.add_argument(
        "--output",
        default=(
            "reports/"
            "benchmark_challenge_documents_expanded_failure_analysis.json"
        ),
        help="Path for saving the failure analysis JSON.",
    )

    args = parser.parse_args()

    report_path = Path(args.report)
    output_path = Path(args.output)

    if not report_path.exists():
        parser.error(f"Benchmark report not found: {report_path}")

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        analysis = analyze_retrieval_failures(report)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    output = {
        "source_report": str(report_path),
        "retrievers": analysis,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== RAG Retrieval Failure Analysis ===")
    print(f"Source report: {report_path}")
    print(f"Saved analysis: {output_path}")

    for retriever_name, result in analysis.items():
        print(f"\nRetriever: {retriever_name}")
        print(f"Total questions: {result['total_questions']}")
        print(f"Top-1 successes: {result['successes']}")
        print(f"Ranking errors: {result['ranking_errors']}")
        print(f"Retrieval misses: {result['retrieval_misses']}")
        print(f"Top-1 success rate: {result['top1_success_rate']:.2%}")

        if result["failures"]:
            print("\nFailed questions:")

            for failure in result["failures"]:
                print(
                    f"- {failure['sample_id']} | "
                    f"Category: {failure['category']} | "
                    f"Status: {failure['status']} | "
                    f"Expected: {failure['expected_source_ids']} | "
                    f"Retrieved: {failure['retrieved_source_ids']}"
                )


if __name__ == "__main__":
    main()
