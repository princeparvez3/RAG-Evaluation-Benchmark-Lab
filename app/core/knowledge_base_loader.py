
import json
from pathlib import Path

from pydantic import ValidationError

from app.models.document import KnowledgeDocument


class KnowledgeBaseLoadError(ValueError):
    """Raised when the knowledge base cannot be loaded safely."""


def load_knowledge_base(
    path: str | Path,
) -> list[KnowledgeDocument]:
    """Load and validate knowledge documents from a JSONL file."""

    file_path = Path(path)

    if not file_path.is_file():
        raise KnowledgeBaseLoadError(
            f"Knowledge base file not found: {file_path}"
        )

    documents: list[KnowledgeDocument] = []
    seen_ids: set[str] = set()

    try:
        with file_path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                if not raw_line.strip():
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise KnowledgeBaseLoadError(
                        f"Invalid JSON on line {line_number}: {exc.msg}"
                    ) from exc

                if not isinstance(record, dict):
                    raise KnowledgeBaseLoadError(
                        f"Line {line_number} must contain a JSON object."
                    )

                try:
                    document = KnowledgeDocument.model_validate(record)
                except ValidationError as exc:
                    raise KnowledgeBaseLoadError(
                        f"Invalid document on line {line_number}: {exc}"
                    ) from exc

                if document.source_id in seen_ids:
                    raise KnowledgeBaseLoadError(
                        f"Duplicate source_id '{document.source_id}' "
                        f"on line {line_number}."
                    )

                seen_ids.add(document.source_id)
                documents.append(document)

    except UnicodeDecodeError as exc:
        raise KnowledgeBaseLoadError(
            f"Knowledge base must use UTF-8 encoding: {file_path}"
        ) from exc

    if not documents:
        raise KnowledgeBaseLoadError(
            "The knowledge base contains no documents."
        )

    return documents
