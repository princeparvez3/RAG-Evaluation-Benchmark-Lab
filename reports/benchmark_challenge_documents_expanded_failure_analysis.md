# RAG Retrieval Failure Analysis

## Overview

- **Source report:** `reports\benchmark_challenge_documents_expanded_retriever_comparison.json`
- **Retrievers evaluated:** 2

## Performance Summary

| Retriever | Questions | Top-1 Successes | Ranking Errors | Retrieval Misses | Top-1 Success Rate |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 15 | 12 | 3 | 0 | 80.00% |
| BM25 | 15 | 11 | 3 | 1 | 73.33% |

## TF-IDF

- **Top-1 success rate:** 80.00%
- **Ranking errors:** 3
- **Retrieval misses:** 0

### Category Breakdown

| Category | Outcome | Count |
|---|---|---:|
| cost | RANKING_ERROR | 1 |
| cost | SUCCESS | 2 |
| limits | RANKING_ERROR | 1 |
| limits | SUCCESS | 2 |
| policy | SUCCESS | 3 |
| product_faq | SUCCESS | 3 |
| security | RANKING_ERROR | 1 |
| security | SUCCESS | 2 |

### Failed Questions

#### CH-008

- **Category:** limits
- **Status:** RANKING_ERROR
- **Expected sources:** `doc-limits-01`
- **Retrieved sources:** `doc-template-limit-01, doc-limits-01, doc-document-limit-01`

#### CH-011

- **Category:** security
- **Status:** RANKING_ERROR
- **Expected sources:** `doc-security-01`
- **Retrieved sources:** `doc-secret-rotation-01, doc-security-01, doc-api-rate-01`

#### CH-014

- **Category:** cost
- **Status:** RANKING_ERROR
- **Expected sources:** `doc-cost-01`
- **Retrieved sources:** `doc-cost-dashboard-01, doc-cost-01, doc-pricing-01`


## BM25

- **Top-1 success rate:** 73.33%
- **Ranking errors:** 3
- **Retrieval misses:** 1

### Category Breakdown

| Category | Outcome | Count |
|---|---|---:|
| cost | RANKING_ERROR | 1 |
| cost | SUCCESS | 2 |
| limits | RANKING_ERROR | 1 |
| limits | RETRIEVAL_MISS | 1 |
| limits | SUCCESS | 1 |
| policy | SUCCESS | 3 |
| product_faq | SUCCESS | 3 |
| security | RANKING_ERROR | 1 |
| security | SUCCESS | 2 |

### Failed Questions

#### CH-008

- **Category:** limits
- **Status:** RETRIEVAL_MISS
- **Expected sources:** `doc-limits-01`
- **Retrieved sources:** `doc-template-limit-01, doc-document-limit-01, doc-api-rate-01`

#### CH-009

- **Category:** limits
- **Status:** RANKING_ERROR
- **Expected sources:** `doc-limits-01`
- **Retrieved sources:** `doc-api-rate-01, doc-document-limit-01, doc-limits-01`

#### CH-011

- **Category:** security
- **Status:** RANKING_ERROR
- **Expected sources:** `doc-security-01`
- **Retrieved sources:** `doc-import-01, doc-security-01, doc-secret-rotation-01`

#### CH-014

- **Category:** cost
- **Status:** RANKING_ERROR
- **Expected sources:** `doc-cost-01`
- **Retrieved sources:** `doc-pricing-01, doc-cost-dashboard-01, doc-cost-01`

## Interpretation

A ranking error means an expected source was retrieved, but not at rank 1.

A retrieval miss means none of the retrieved sources matched the expected sources.

These results come from a small, synthetic benchmark. They should not be treated as proof that one retriever will outperform another on every dataset.
