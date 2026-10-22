
from pydantic import BaseModel, Field

from app.models.document import KnowledgeDocument


class RetrievedDocument(BaseModel):
    """A retrieved knowledge document with its relevance score."""

    document: KnowledgeDocument
    score: float = Field(ge=0.0, le=1.0)


class BM25RetrievedDocument(BaseModel):
    """A BM25 result with its raw retrieval score."""

    document: KnowledgeDocument
    score: float
