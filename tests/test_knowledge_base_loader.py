
import json
from pathlib import Path

import pytest

from app.core.knowledge_base_loader import (
    KnowledgeBaseLoadError,
    load_knowledge_base,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_project_knowledge_base():
    path = (
        PROJECT_ROOT
        / "data"
        / "knowledge_base"
        / "documents.jsonl"
    )

    documents = load_knowledge_base(path)

    assert len(documents) == 5
    assert documents[0].source_id == "doc-export-01"
    assert "CSV, JSONL, and Parquet" in documents[0].content


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(KnowledgeBaseLoadError, match="not found"):
        load_knowledge_base(tmp_path / "missing.jsonl")


def test_invalid_json_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(KnowledgeBaseLoadError, match="line 1"):
        load_knowledge_base(path)


def test_duplicate_source_ids_are_rejected(tmp_path):
    record = {
        "source_id": "doc-001",
        "title": "First document",
        "content": "First content.",
    }
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        KnowledgeBaseLoadError,
        match="Duplicate source_id",
    ):
        load_knowledge_base(path)


def test_empty_content_is_rejected(tmp_path):
    record = {
        "source_id": "doc-001",
        "title": "Empty document",
        "content": "",
    }
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(KnowledgeBaseLoadError, match="line 1"):
        load_knowledge_base(path)


def test_empty_knowledge_base_is_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n   \n", encoding="utf-8")

    with pytest.raises(
        KnowledgeBaseLoadError,
        match="no documents",
    ):
        load_knowledge_base(path)
