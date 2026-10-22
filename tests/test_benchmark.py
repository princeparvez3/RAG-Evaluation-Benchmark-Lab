
import pytest
from pydantic import ValidationError

from app.models.benchmark import BenchmarkSample, ModelResponse


def test_valid_benchmark_sample():
    sample = BenchmarkSample(
        sample_id="RAG-001",
        question="What is retrieval-augmented generation?",
        reference_answer="RAG retrieves relevant information to help generate an answer.",
        retrieved_contexts=["RAG combines retrieval with answer generation."],
        expected_citations=["source-1"],
        category="technical",
        metadata={"difficulty": "easy"},
    )

    assert sample.sample_id == "RAG-001"
    assert sample.category == "technical"
    assert len(sample.retrieved_contexts) == 1


def test_benchmark_defaults():
    sample = BenchmarkSample(
        sample_id="RAG-002",
        question="What is a benchmark?",
        reference_answer="A benchmark is a standard used for comparison.",
    )

    assert sample.retrieved_contexts == []
    assert sample.expected_citations == []
    assert sample.category == "general"
    assert sample.metadata == {}


def test_empty_question_is_rejected():
    with pytest.raises(ValidationError):
        BenchmarkSample(
            sample_id="RAG-003",
            question="",
            reference_answer="A sample reference answer.",
        )


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        BenchmarkSample(
            sample_id="RAG-004",
            question="What is RAG?",
            reference_answer="RAG uses retrieved information.",
            unexpected_field="not allowed",
        )


def test_valid_model_response():
    response = ModelResponse(
        sample_id="RAG-001",
        model_name="baseline-model",
        generated_answer="RAG uses retrieved documents to inform its answer.",
        retrieved_contexts=["A retrieved supporting passage."],
        citations=["source-1"],
        latency_ms=125.5,
        input_tokens=100,
        output_tokens=35,
        estimated_cost_usd=0.001,
    )

    assert response.model_name == "baseline-model"
    assert response.latency_ms == 125.5
    assert response.estimated_cost_usd == 0.001


def test_negative_latency_is_rejected():
    with pytest.raises(ValidationError):
        ModelResponse(
            sample_id="RAG-001",
            model_name="baseline-model",
            generated_answer="Example answer.",
            latency_ms=-10,
        )


def test_negative_cost_is_rejected():
    with pytest.raises(ValidationError):
        ModelResponse(
            sample_id="RAG-001",
            model_name="baseline-model",
            generated_answer="Example answer.",
            estimated_cost_usd=-0.5,
        )
