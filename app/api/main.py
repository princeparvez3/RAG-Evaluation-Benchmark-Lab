
from fastapi import FastAPI

app = FastAPI(
    title="RAG Evaluation & Benchmark Lab API",
    description=(
        "API for the RAG retrieval benchmarking and evaluation project."
    ),
    version="1.0.0",
)


@app.get("/", tags=["Health"])
def root():
    return {
        "message": "RAG Evaluation & Benchmark Lab API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}


@app.get("/api/v1/info", tags=["Info"])
def project_info():
    return {
        "project": "RAG Evaluation & Benchmark Lab",
        "version": "1.0.0",
        "retrievers": ["TF-IDF", "BM25"],
        "metrics": ["Precision@K", "Recall@K", "MRR@K"],
    }
