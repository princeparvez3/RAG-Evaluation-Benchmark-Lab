import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Export retrieval failure analysis as Markdown."
    )

    parser.add_argument(
        "--input",
        default=(
            "reports/"
            "benchmark_challenge_documents_expanded_failure_analysis.json"
        ),
        help="Path to the failure analysis JSON.",
    )

    parser.add_argument(
        "--output",
        default=(
            "reports/"
            "benchmark_challenge_documents_expanded_failure_analysis.md"
        ),
        help="Path for the Markdown report.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        parser.error(f"Input report not found: {input_path}")

    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"Could not read input report: {error}")

    retrievers = data.get("retrievers")

    if not isinstance(retrievers, dict) or not retrievers:
        parser.error("Input report must contain retriever analysis.")

    lines = [
        "# RAG Retrieval Failure Analysis",
        "",
        "## Overview",
        "",
        f"- **Source report:** `{data.get('source_report', 'Unknown')}`",
        f"- **Retrievers evaluated:** {len(retrievers)}",
        "",
        "## Performance Summary",
        "",
        "| Retriever | Questions | Top-1 Successes | Ranking Errors "
        "| Retrieval Misses | Top-1 Success Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, result in retrievers.items():
        lines.append(
            f"| {name} "
            f"| {result['total_questions']} "
            f"| {result['successes']} "
            f"| {result['ranking_errors']} "
            f"| {result['retrieval_misses']} "
            f"| {result['top1_success_rate']:.2%} |"
        )

    for name, result in retrievers.items():
        lines.extend([
            "",
            f"## {name}",
            "",
            f"- **Top-1 success rate:** "
            f"{result['top1_success_rate']:.2%}",
            f"- **Ranking errors:** {result['ranking_errors']}",
            f"- **Retrieval misses:** {result['retrieval_misses']}",
            "",
            "### Category Breakdown",
            "",
            "| Category | Outcome | Count |",
            "|---|---|---:|",
        ])

        for item in result.get("category_summary", []):
            lines.append(
                f"| {item['category']} "
                f"| {item['status']} "
                f"| {item['count']} |"
            )

        lines.extend(["", "### Failed Questions", ""])

        failures = result.get("failures", [])

        if not failures:
            lines.append("No failed questions.")
        else:
            for failure in failures:
                lines.extend([
                    f"#### {failure['sample_id']}",
                    "",
                    f"- **Category:** {failure['category']}",
                    f"- **Status:** {failure['status']}",
                    f"- **Expected sources:** "
                    f"`{', '.join(failure['expected_source_ids'])}`",
                    f"- **Retrieved sources:** "
                    f"`{', '.join(failure['retrieved_source_ids'])}`",
                    "",
                ])

    lines.extend([
        "## Interpretation",
        "",
        "A ranking error means an expected source was retrieved, "
        "but not at rank 1.",
        "",
        "A retrieval miss means none of the retrieved sources matched "
        "the expected sources.",
        "",
        "These results come from a small, synthetic benchmark. "
        "They should not be treated as proof that one retriever "
        "will outperform another on every dataset.",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Markdown report generated successfully.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()