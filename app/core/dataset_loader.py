
import json
from pathlib import Path

from pydantic import ValidationError

from app.models.benchmark import BenchmarkSample


class DatasetLoadError(ValueError):
    """Raised when a benchmark dataset cannot be loaded safely."""


def load_benchmark_dataset(
    path: str | Path,
) -> list[BenchmarkSample]:
    """Load and validate benchmark records from a JSONL file."""

    file_path = Path(path)

    if not file_path.is_file():
        raise DatasetLoadError(
            f"Dataset file not found: {file_path}"
        )

    samples: list[BenchmarkSample] = []
    seen_ids: set[str] = set()

    try:
        with file_path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                if not raw_line.strip():
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise DatasetLoadError(
                        f"Invalid JSON on line {line_number}: {exc.msg}"
                    ) from exc

                if not isinstance(record, dict):
                    raise DatasetLoadError(
                        f"Line {line_number} must contain a JSON object."
                    )

                try:
                    sample = BenchmarkSample.model_validate(record)
                except ValidationError as exc:
                    raise DatasetLoadError(
                        f"Invalid benchmark sample on line {line_number}: "
                        f"{exc}"
                    ) from exc

                if sample.sample_id in seen_ids:
                    raise DatasetLoadError(
                        f"Duplicate sample_id '{sample.sample_id}' "
                        f"on line {line_number}."
                    )

                seen_ids.add(sample.sample_id)
                samples.append(sample)

    except UnicodeDecodeError as exc:
        raise DatasetLoadError(
            f"Dataset must use UTF-8 encoding: {file_path}"
        ) from exc

    if not samples:
        raise DatasetLoadError("The benchmark dataset contains no records.")

    return samples
