
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class BenchmarkSample(BaseModel):
    """One test case used to evaluate a RAG system."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(
        ..., min_length=1, description="Unique benchmark sample ID"
    )
    question: str = Field(
        ..., min_length=1, description="Question asked by the user"
    )
    reference_answer: str = Field(
        ..., min_length=1, description="Expected or reference answer"
    )
    retrieved_contexts: list[str] = Field(
        default_factory=list,
        description="Documents or passages retrieved by the RAG system",
    )
    expected_citations: list[str] = Field(
        default_factory=list,
        description="Expected source IDs or citation references",
    )
    category: str = Field(
        default="general",
        description="Benchmark category, such as factual or technical",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional string-based benchmark metadata",
    )


class ModelResponse(BaseModel):
    """A model's response to one benchmark question."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    generated_answer: str = Field(..., min_length=1)
    retrieved_contexts: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    latency_ms: Optional[float] = Field(default=None, ge=0)
    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)
    estimated_cost_usd: Optional[float] = Field(default=None, ge=0)
