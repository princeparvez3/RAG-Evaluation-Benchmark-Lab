
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.document import KnowledgeDocument
from app.models.retrieval import RetrievedDocument


class TfidfRetriever:
    """Retrieve relevant documents using TF-IDF and cosine similarity."""

    def __init__(self, documents: list[KnowledgeDocument]) -> None:
        if not documents:
            raise ValueError("At least one knowledge document is required.")

        self.documents = list(documents)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        document_texts = [
            f"{doc.title} {doc.category} {doc.content}"
            for doc in self.documents
        ]

        self.document_matrix = self.vectorizer.fit_transform(
            document_texts
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedDocument]:
        """Return up to top_k matching documents in relevance order."""

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

        query_vector = self.vectorizer.transform([query])

        # No vocabulary overlap means no document can be matched.
        if query_vector.nnz == 0:
            return []

        scores = cosine_similarity(
            query_vector,
            self.document_matrix,
        ).ravel()

        # Sort by descending score; original order breaks ties.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), index),
        )

        results: list[RetrievedDocument] = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                break

            # Guard against floating-point rounding at the upper bound.
            score = min(max(score, 0.0), 1.0)

            results.append(
                RetrievedDocument(
                    document=self.documents[index],
                    score=score,
                )
            )

            if len(results) >= top_k:
                break

        return results
