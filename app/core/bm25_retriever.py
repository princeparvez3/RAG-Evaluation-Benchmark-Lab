
import re

from rank_bm25 import BM25Okapi

from app.models.document import KnowledgeDocument
from app.models.retrieval import BM25RetrievedDocument


def tokenize(text: str) -> list[str]:
    """Convert text to lowercase word tokens."""

    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BM25Retriever:
    """Retrieve documents using BM25 lexical relevance ranking."""

    def __init__(self, documents: list[KnowledgeDocument]) -> None:
        if not documents:
            raise ValueError(
                "At least one knowledge document is required."
            )

        self.documents = list(documents)

        self.tokenized_corpus = [
            tokenize(f"{doc.title} {doc.category} {doc.content}")
            for doc in self.documents
        ]

        if not any(self.tokenized_corpus):
            raise ValueError(
                "Knowledge documents contain no searchable words."
            )

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[BM25RetrievedDocument]:
        """Return matching documents ranked by descending BM25 score."""

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer.")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        query_terms = set(query_tokens)

        # Exclude documents with no lexical overlap with the query.
        matching_indices = [
            index
            for index, document_tokens in enumerate(
                self.tokenized_corpus
            )
            if query_terms.intersection(document_tokens)
        ]

        if not matching_indices:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            matching_indices,
            key=lambda index: (-float(scores[index]), index),
        )

        return [
            BM25RetrievedDocument(
                document=self.documents[index],
                score=float(scores[index]),
            )
            for index in ranked_indices[:top_k]
        ]
