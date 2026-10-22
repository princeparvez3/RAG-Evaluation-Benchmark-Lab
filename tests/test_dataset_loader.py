
import json
from pathlib import Path

import pytest

from app.core.dataset_loader import (
    DatasetLoadError,
    load_benchmark_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_project_benchmark_dataset():
    path = PROJECT_ROOT / "data" / "samples" / "benchmark.jsonl"

    samples = load_benchmark_dataset(path)

    assert len(samples) == 5
    assert samples[0].sample_id == "RAG-001"
    assert samples[0].expected_citations == ["doc-export-01"]


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(DatasetLoadError, match="not found"):
        load_benchmark_dataset(tmp_path / "missing.jsonl")



def test_malformed_json_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"

    valid_record = {
        "sample_id": "RAG-001",
        "question": "What is RAG?",
        "reference_answer": "RAG uses retrieved information to generate answers.",
    }

    path.write_text(
        json.dumps(valid_record) + "\n{invalid json}\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetLoadError, match="line 2"):
        load_benchmark_dataset(path)



def test_duplicate_sample_ids_are_rejected(tmp_path):
    record = {
        "sample_id": "RAG-001",
        "question": "What is RAG?",
        "reference_answer": "RAG retrieves context for generation.",
    }
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetLoadError, match="Duplicate sample_id"):
        load_benchmark_dataset(path)


def test_invalid_record_is_rejected(tmp_path):
    record = {
        "sample_id": "RAG-001",
        "question": "",
        "reference_answer": "A reference answer.",
    }
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="line 1"):
        load_benchmark_dataset(path)


def test_empty_dataset_is_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n   \n", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="no records"):
        load_benchmark_dataset(path)
