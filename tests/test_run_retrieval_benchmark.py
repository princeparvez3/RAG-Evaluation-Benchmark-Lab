
import json

import pytest

from scripts.run_retrieval_benchmark import run_benchmark


def test_benchmark_generates_json_report(tmp_path):
    output_path = tmp_path / "reports" / "benchmark.json"

    report, saved_path = run_benchmark(
        k=3,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert saved_path.is_file()
    assert report["total_samples"] == 5
    assert report["retriever"] == "TF-IDF"
    assert report["evaluation_type"] == "retrieval_only"

    saved_report = json.loads(
        saved_path.read_text(encoding="utf-8")
    )

    assert saved_report["total_samples"] == 5
    assert len(saved_report["results"]) == 5
    assert "mrr_at_k" in saved_report["aggregate_metrics"]


def test_benchmark_rejects_invalid_k(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        run_benchmark(k=0, output_path=tmp_path / "report.json")
