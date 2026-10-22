
from pydantic import BaseModel, ConfigDict, Field


class KnowledgeDocument(BaseModel):
    """A source document available to the RAG retrieval system."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    category: str = Field(default="general", min_length=1)
