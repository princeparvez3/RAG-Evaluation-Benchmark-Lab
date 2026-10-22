# RAG Evaluation & Benchmark Lab

A Python-based benchmarking project for evaluating and comparing retrieval algorithms used in Retrieval-Augmented Generation (RAG) systems.

The project compares TF-IDF and BM25 retrievers, measures retrieval performance, identifies ranking errors and retrieval misses, and generates machine-readable and human-readable evaluation reports.

## Project Overview

Retrieval quality is an important part of a RAG system. A relevant document must not only be retrieved but should ideally appear near the top of the results.

This project provides a practical evaluation workflow for comparing retrieval algorithms on a controlled benchmark dataset.

## Key Features

- **Multiple retrievers:** Compare TF-IDF and BM25.
- **Retrieval metrics:** Evaluate Precision@K, Recall@K, and Mean Reciprocal Rank (MRR@K).
- **Benchmark datasets:** Run evaluations using benchmark questions and knowledge-base documents.
- **Challenge dataset:** Test 15 synthetic paraphrased questions across export, retention, limits, security, and cost categories.
- **Expanded knowledge base:** Evaluate retrieval behavior against 15 documents, including distractor documents.
- **Failure analysis:** Identify top-1 successes, ranking errors, and retrieval misses.
- **Report generation:** Export comparison results and failure analysis in JSON and Markdown.
- **Automated testing:** Validate project behavior with pytest.

## Technology Stack

- Python
- TF-IDF retrieval
- BM25 using `rank-bm25`
- JSON and JSONL
- pytest

## Project Structure

```text
RAG-Evaluation-Benchmark-Lab/
├── app/
│   ├── core/
│   │   ├── dataset_loader.py
│   │   ├── knowledge_base_loader.py
│   │   ├── retriever.py
│   │   └── bm25_retriever.py
│   ├── metrics/
│   │   └── retrieval_metrics.py
│   └── services/
│       ├── retrieval_evaluator.py
│       └── failure_analyzer.py
├── data/
│   ├── samples/
│   │   ├── benchmark.jsonl
│   │   └── benchmark_challenge.jsonl
│   └── knowledge_base/
│       ├── documents.jsonl
│       └── documents_expanded.jsonl
├── reports/
├── scripts/
│   ├── run_retrieval_benchmark.py
│   ├── compare_retrievers.py
│   ├── analyze_retrieval_failures.py
│   └── export_failure_analysis_markdown.py
├── tests/
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and open its directory:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG-Evaluation-Benchmark-Lab
```

Create and activate a virtual environment.

**Windows:**

```bat
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bat
python -m pip install -r requirements.txt
```

## Run the Retriever Comparison

Run the challenge benchmark against the expanded knowledge base:

```bat
python -m scripts.compare_retrievers --dataset data/samples/benchmark_challenge.jsonl --knowledge-base data/knowledge_base/documents_expanded.jsonl
```

The comparison evaluates both retrievers and saves a JSON report in the `reports/` directory.

## Analyze Retrieval Failures

Generate failure analysis from the comparison report:

```bat
python -m scripts.analyze_retrieval_failures
```

This identifies three outcome types:

- **SUCCESS:** An expected source is ranked first.
- **RANKING_ERROR:** An expected source is retrieved, but not ranked first.
- **RETRIEVAL_MISS:** No expected source appears in the retrieved results.

## Export a Markdown Report

Convert the failure analysis JSON into a readable Markdown report:

```bat
python -m scripts.export_failure_analysis_markdown
```

Generated reports are saved under `reports/`.

## Benchmark Results

### Challenge Benchmark

Evaluation configuration:

- Questions: 15
- Knowledge-base documents: 15
- Retrieval cutoff: K = 3

| Metric | TF-IDF | BM25 |
|---|---:|---:|
| Precision@3 | 0.3333 | 0.3111 |
| Recall@3 | 1.0000 | 0.9333 |
| MRR@3 | 0.9000 | 0.8111 |
| Top-1 success rate | 80.00% | 73.33% |
| Ranking errors | 3 | 3 |
| Retrieval misses | 0 | 1 |

### Interpretation

On this particular synthetic challenge benchmark, TF-IDF achieved a higher MRR@3 and top-1 success rate than BM25.

However, the benchmark is small and synthetic. These results do not establish that TF-IDF is universally better than BM25. Performance should be re-evaluated on larger, more diverse datasets before drawing broader conclusions.

## Run Tests

Run the complete test suite:

```bat
python -m pytest tests -v
```

**Latest verified result:** 68 tests passed.

## Skills Demonstrated

- Information retrieval and retrieval benchmarking
- Evaluation metric implementation
- Dataset and knowledge-base handling
- Failure analysis and error categorization
- JSON/JSONL data processing
- Python modular architecture
- Automated testing with pytest
- Technical reporting and experiment analysis

## Future Improvements

- Evaluate additional retrieval methods.
- Experiment with configurable tokenization and ranking parameters.
- Add larger and more diverse benchmark datasets.
- Integrate semantic and embedding-based retrieval.
- Extend the evaluation workflow to measure RAG answer quality and groundedness.
- Build a dashboard for comparing benchmark runs.

## Project Scope

This project currently focuses on **retrieval evaluation**, rather than a complete production RAG application. Its results are intended for controlled experimentation and learning.
