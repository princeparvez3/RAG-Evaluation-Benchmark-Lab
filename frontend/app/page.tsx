
"use client";

import { useState, type ReactNode } from "react";
import RunHistory from "./components/run-history";

const API_BASE = "http://127.0.0.1:8000";

type RetrieverName = "TF-IDF" | "BM25";
type DatasetName = "default" | "expanded";

interface AggregateMetrics {
  precision_at_k: number;
  recall_at_k: number;
  mrr_at_k: number;
}

interface QuestionResult {
  sample_id: string;
  question: string;
  category: string;
  expected_source_ids: string[];
  retrieved_source_ids: string[];
  retrieved_documents: {
    source_id: string;
    title: string;
    score: number;
  }[];
  precision_at_k: number;
  recall_at_k: number;
  reciprocal_rank_at_k: number;
}

interface RetrieverResult {
  aggregate_metrics: AggregateMetrics;
  per_question: QuestionResult[];
}

interface BenchmarkReport {
  evaluation_type: string;
  dataset: string;
  knowledge_base: string;
  total_samples: number;
  knowledge_base_document_count: number;
  k: number;
  retrievers: Record<RetrieverName, RetrieverResult>;
}

interface BenchmarkResponse {
  status: string;
  report_file: string;
  report: BenchmarkReport;
}

interface FailureOutcome {
  sample_id: string;
  category: string;
  expected_source_ids: string[];
  retrieved_source_ids: string[];
  status: "SUCCESS" | "RANKING_ERROR" | "RETRIEVAL_MISS";
}

interface CategorySummary {
  category: string;
  status: string;
  count: number;
}

interface RetrieverFailureAnalysis {
  total_questions: number;
  successes: number;
  ranking_errors: number;
  retrieval_misses: number;
  top1_success_rate: number;
  failures: FailureOutcome[];
  category_summary: CategorySummary[];
}

interface FailureResponse {
  status: string;
  source_report: string;
  retrievers: Record<RetrieverName, RetrieverFailureAnalysis>;
}

const navigation = [
  "Overview",
  "Benchmarks",
  "Retrievers",
  "Evaluation Runs",
  "Failure Analysis",
  "Reports",
];

const icons: Record<string, ReactNode> = {
  Overview: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1" />
      <rect x="14" y="3" width="7" height="7" rx="1" />
      <rect x="3" y="14" width="7" height="7" rx="1" />
      <rect x="14" y="14" width="7" height="7" rx="1" />
    </>
  ),
  Benchmarks: (
    <>
      <path d="M3 3v18h18" />
      <path d="m7 14 4-4 4 3 6-7" />
    </>
  ),
  Retrievers: (
    <>
      <path d="m12 3 9 5-9 5-9-5 9-5Z" />
      <path d="m3 12 9 5 9-5M3 16l9 5 9-5" />
    </>
  ),
  "Evaluation Runs": (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </>
  ),
  "Failure Analysis": (
    <>
      <circle cx="10.5" cy="10.5" r="6.5" />
      <path d="m16 16 4.5 4.5" />
    </>
  ),
  Reports: (
    <>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" />
      <path d="M14 2v6h6M8 13h8M8 17h8" />
    </>
  ),
  server: (
    <>
      <rect x="3" y="4" width="18" height="7" rx="2" />
      <rect x="3" y="14" width="18" height="7" rx="2" />
      <path d="M7 7.5h.01M7 17.5h.01M11 7.5h6M11 17.5h6" />
    </>
  ),
  check: <path d="m5 12 4 4L19 6" />,
};

function Icon({
  name,
  size = 18,
}: {
  name: string;
  size?: number;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {icons[name] ?? icons.Overview}
    </svg>
  );
}

function MetricBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number | undefined;
  color: string;
}) {
  const valid =
    typeof value === "number" && Number.isFinite(value);

  const safeValue = valid
    ? Math.min(1, Math.max(0, value))
    : 0;

  return (
    <div className="mt-4">
      <div className="mb-2 flex items-center justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-slate-200">
          {valid ? value.toFixed(3) : "—"}
        </span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-white/[0.06]">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${safeValue * 100}%` }}
        />
      </div>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  const success = status === "SUCCESS";

  return (
    <span
      className={`inline-flex whitespace-nowrap rounded-full px-2.5 py-1 text-[10px] font-medium ${
        success
          ? "bg-emerald-300/10 text-emerald-200"
          : "bg-rose-300/10 text-rose-200"
      }`}
    >
      {status.replaceAll("_", " ")}
    </span>
  );
}

export default function Home() {
  const [activeView, setActiveView] = useState("Overview");

  const [selectedDataset, setSelectedDataset] =
    useState<DatasetName>("expanded");

  const [benchmark, setBenchmark] =
    useState<BenchmarkResponse | null>(null);

  const [failureAnalysis, setFailureAnalysis] =
    useState<FailureResponse | null>(null);

  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [failureLoading, setFailureLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const report = benchmark?.report;
  const tfidf = report?.retrievers["TF-IDF"];
  const bm25 = report?.retrievers.BM25;

  async function runBenchmark() {
    setBenchmarkLoading(true);
    setError("");
    setNotice("");
    setFailureAnalysis(null);

    try {
      const response = await fetch(
        `${API_BASE}/api/v1/benchmark/run?k=3&dataset=${selectedDataset}`,
        { method: "POST" }
      );

      const body = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          body.detail || `Request failed: ${response.status}`
        );
      }

      if (
        !body.report?.retrievers?.["TF-IDF"] ||
        !body.report?.retrievers?.BM25
      ) {
        throw new Error("Unexpected benchmark response from API.");
      }

      setBenchmark(body as BenchmarkResponse);

      setNotice(
        `Benchmark completed successfully using the ${
          selectedDataset === "expanded"
            ? "Expanded Dataset"
            : "Default Dataset"
        }.`
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not run benchmark."
      );
    } finally {
      setBenchmarkLoading(false);
    }
  }

  async function loadFailureAnalysis() {
    setFailureLoading(true);
    setError("");
    setNotice("");

    try {
      const response = await fetch(
        `${API_BASE}/api/v1/benchmark/failures`
      );

      const body = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          body.detail || `Request failed: ${response.status}`
        );
      }

      if (
        !body.retrievers?.["TF-IDF"] ||
        !body.retrievers?.BM25
      ) {
        throw new Error(
          "Unexpected failure-analysis response from API."
        );
      }

      setFailureAnalysis(body as FailureResponse);

      setNotice(
        "Failure analysis loaded from the latest saved benchmark report."
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not load failure analysis."
      );
    } finally {
      setFailureLoading(false);
    }
  }

  function downloadReport() {
    if (!benchmark) return;

    const blob = new Blob(
      [JSON.stringify(benchmark, null, 2)],
      { type: "application/json" }
    );

    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");

    anchor.href = url;
    anchor.download = "rag-retriever-comparison.json";
    anchor.click();

    URL.revokeObjectURL(url);
  }

  function downloadFailureAnalysis() {
    if (!failureAnalysis) return;

    const blob = new Blob(
      [JSON.stringify(failureAnalysis, null, 2)],
      { type: "application/json" }
    );

    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");

    anchor.href = url;
    anchor.download = "rag-failure-analysis.json";
    anchor.click();

    URL.revokeObjectURL(url);
  }

  const metricCards = [
    {
      label: "Benchmark questions",
      value: report ? String(report.total_samples) : "—",
      detail: report
        ? `Dataset: ${report.dataset}`
        : "Run benchmark to load",
      icon: "Overview",
      color: "text-cyan-200",
      bg: "bg-cyan-300/10",
    },
    {
      label: "Knowledge base",
      value: report
        ? String(report.knowledge_base_document_count)
        : "—",
      detail: report
        ? report.knowledge_base
        : "Document count from API",
      icon: "Reports",
      color: "text-violet-200",
      bg: "bg-violet-300/10",
    },
    {
      label: "Retrieval methods",
      value: report
        ? String(Object.keys(report.retrievers).length)
        : "2",
      detail: "TF-IDF and BM25",
      icon: "Retrievers",
      color: "text-emerald-200",
      bg: "bg-emerald-300/10",
    },
    {
      label: "Evaluation metrics",
      value: "3",
      detail: "Precision@K · Recall@K · MRR@K",
      icon: "Benchmarks",
      color: "text-amber-200",
      bg: "bg-amber-300/10",
    },
  ];

  return (
    <main className="min-h-screen bg-[#0a0d14] text-slate-100">
      <div className="flex min-h-screen flex-col">
        {/* TOP NAVIGATION */}
        <header className="border-b border-white/[0.07] bg-[#0d111b] px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-400 text-slate-950">
                <Icon name="Retrievers" size={22} />
              </div>

              <div>
                <p className="text-sm font-bold tracking-wide">
                  RAG LAB
                </p>
                <p className="mt-0.5 text-[11px] text-slate-500">
                  Evaluation Studio
                </p>
              </div>

              <span className="ml-2 hidden rounded-full border border-white/[0.08] px-2.5 py-1 text-[10px] text-slate-400 sm:inline-flex">
                Local preview
              </span>
            </div>

            <nav
              aria-label="Main navigation"
              className="flex gap-1 overflow-x-auto pb-1 xl:pb-0"
            >
              {navigation.map((item) => {
                const active = activeView === item;

                return (
                  <button
                    key={item}
                    onClick={() => {
                      setActiveView(item);
                      setError("");
                      setNotice("");
                    }}
                    aria-current={active ? "page" : undefined}
                    className={`flex shrink-0 items-center gap-2 whitespace-nowrap rounded-lg border px-3 py-2.5 text-xs transition ${
                      active
                        ? "border-cyan-400/15 bg-cyan-400/10 font-medium text-cyan-300"
                        : "border-transparent text-slate-400 hover:bg-white/[0.04] hover:text-slate-100"
                    }`}
                  >
                    <Icon name={item} size={16} />
                    {item}
                  </button>
                );
              })}
            </nav>
          </div>
        </header>

        {/* PAGE HEADER */}
        <section className="min-w-0 flex-1">
          <header className="flex min-h-[74px] items-center justify-between border-b border-white/[0.07] px-5 py-4 sm:px-8">
            <div>
              <div className="flex items-center gap-2 text-[11px] text-slate-500">
                <span>RAG LAB</span>
                <span>/</span>
                <span className="text-slate-300">{activeView}</span>
              </div>

              <h1 className="mt-1 text-lg font-semibold tracking-tight sm:text-xl">
                {activeView === "Overview"
                  ? "Benchmark Intelligence"
                  : activeView}
              </h1>
            </div>

            <div className="flex items-center gap-3">
              <span
                className={`hidden items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] sm:inline-flex ${
                  error
                    ? "border-rose-400/20 bg-rose-400/[0.06] text-rose-200"
                    : benchmark
                      ? "border-emerald-400/20 bg-emerald-300/[0.06] text-emerald-200"
                      : "border-amber-400/20 bg-amber-300/[0.06] text-amber-200"
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    error
                      ? "bg-rose-300"
                      : benchmark
                        ? "bg-emerald-300"
                        : "bg-amber-300"
                  }`}
                />

                {error
                  ? "API error"
                  : benchmark
                    ? "Results loaded"
                    : "Not run yet"}
              </span>

              <div className="flex h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-[#171d2a] text-xs font-semibold text-cyan-200">
                PP
              </div>
            </div>
          </header>

          <div className="mx-auto max-w-[1450px] p-4 sm:p-7 lg:p-9">
            {error && (
              <div
                role="alert"
                className="mb-5 rounded-xl border border-rose-400/20 bg-rose-400/[0.06] p-4 text-sm text-rose-200"
              >
                <p className="font-semibold">Request failed</p>
                <p className="mt-2 break-words leading-6">{error}</p>
                <p className="mt-2 text-xs text-rose-200/70">
                  Check that FastAPI is running at 127.0.0.1:8000.
                </p>
              </div>
            )}

            {notice && (
              <div
                role="status"
                className="mb-5 rounded-xl border border-emerald-300/20 bg-emerald-300/[0.06] p-4 text-sm text-emerald-200"
              >
                {notice}
              </div>
            )}

            <div className="mb-7">
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <span className="rounded-md border border-cyan-300/15 bg-cyan-300/[0.07] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-cyan-200">
                  Retrieval evaluation
                </span>

                <span className="text-[11px] text-slate-600">•</span>

                <span className="text-[11px] text-slate-500">
                  Local development environment
                </span>
              </div>

              <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
                {activeView === "Overview" ? (
                  <>
                    Measure retrieval{" "}
                    <span className="text-slate-500">
                      with confidence.
                    </span>
                  </>
                ) : (
                  activeView
                )}
              </h2>

              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
                Evaluate retrieval quality, compare algorithms, and inspect
                the results generated by your Python evaluation pipeline.
              </p>
            </div>

            {/* OVERVIEW */}
            {activeView === "Overview" && (
              <>
                <div className="mb-7 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                  {metricCards.map((card) => (
                    <div
                      key={card.label}
                      className="rounded-xl border border-white/[0.07] bg-[#101520] p-5 transition hover:border-white/[0.13]"
                    >
                      <div className="flex items-center justify-between">
                        <p className="text-xs text-slate-400">
                          {card.label}
                        </p>

                        <div
                          className={`flex h-8 w-8 items-center justify-center rounded-lg ${card.bg} ${card.color}`}
                        >
                          <Icon name={card.icon} size={17} />
                        </div>
                      </div>

                      <p className="mt-5 text-3xl font-semibold tracking-tight">
                        {card.value}
                      </p>

                      <p className="mt-2 text-[11px] text-slate-500">
                        {card.detail}
                      </p>
                    </div>
                  ))}
                </div>

                {/* BENCHMARK CONTROLS — DATASET SELECTOR ADDED */}
                <section className="mb-5 flex flex-col gap-5 rounded-xl border border-white/[0.07] bg-[#101520] p-5 sm:p-6 xl:flex-row xl:items-center xl:justify-between">
                  <div>
                    <h3 className="text-sm font-semibold">
                      Retrieval benchmark
                    </h3>
                    <p className="mt-1 text-xs leading-5 text-slate-500">
                      Select a dataset, then run TF-IDF and BM25 with k = 3.
                    </p>
                  </div>

                  <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
                    <div className="min-w-0 sm:min-w-[240px]">
                      <label
                        htmlFor="dataset-select"
                        className="mb-2 block text-xs font-medium text-slate-400"
                      >
                        Benchmark dataset
                      </label>

                      <select
                        id="dataset-select"
                        value={selectedDataset}
                        onChange={(event) =>
                          setSelectedDataset(
                            event.target.value as DatasetName
                          )
                        }
                        disabled={benchmarkLoading}
                        className="w-full rounded-lg border border-white/10 bg-[#171d2a] px-4 py-3 text-xs text-slate-100 outline-none transition focus:border-cyan-300/50 disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        <option value="expanded">
                          Expanded Dataset — 20 questions
                        </option>
                        <option value="default">
                          Default Dataset — 5 questions
                        </option>
                      </select>

                      <p className="mt-2 text-[10px] leading-4 text-slate-500">
                        {selectedDataset === "expanded"
                          ? "Uses benchmark_expanded.jsonl and 30 knowledge-base documents."
                          : "Uses benchmark.jsonl and the default knowledge base."}
                      </p>
                    </div>

                    <button
                      onClick={() => void runBenchmark()}
                      disabled={benchmarkLoading}
                      className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-cyan-300 px-5 py-3 text-xs font-semibold text-slate-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <Icon name="Benchmarks" size={16} />
                      {benchmarkLoading
                        ? "Running benchmark..."
                        : "Run benchmark"}
                    </button>
                  </div>
                </section>

                <div className="mb-5 grid gap-5 xl:grid-cols-2">
                  {(
                    [
                      {
                        name: "TF-IDF",
                        data: tfidf,
                        color: "bg-cyan-300",
                        description: "TF-IDF with cosine similarity",
                      },
                      {
                        name: "BM25",
                        data: bm25,
                        color: "bg-violet-300",
                        description: "BM25 lexical ranking",
                      },
                    ] as const
                  ).map((item) => (
                    <section
                      key={item.name}
                      className="rounded-xl border border-white/[0.07] bg-[#101520] p-5 sm:p-6"
                    >
                      <h3 className="text-sm font-semibold">
                        {item.name}
                      </h3>
                      <p className="mt-1 text-xs text-slate-500">
                        {item.description}
                      </p>

                      <MetricBar
                        label="Precision@K"
                        value={item.data?.aggregate_metrics.precision_at_k}
                        color={item.color}
                      />
                      <MetricBar
                        label="Recall@K"
                        value={item.data?.aggregate_metrics.recall_at_k}
                        color={item.color}
                      />
                      <MetricBar
                        label="MRR@K"
                        value={item.data?.aggregate_metrics.mrr_at_k}
                        color={item.color}
                      />

                      {!report && (
                        <p className="mt-5 text-xs text-slate-600">
                          Run a benchmark to display measured results.
                        </p>
                      )}
                    </section>
                  ))}
                </div>

                <section className="overflow-hidden rounded-xl border border-white/[0.07] bg-[#101520]">
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.06] px-5 py-5 sm:px-6">
                    <div>
                      <h3 className="text-sm font-semibold">
                        Per-question retrieval results
                      </h3>
                      <p className="mt-1 text-xs text-slate-500">
                        Expected sources compared with the first retrieved result.
                      </p>
                    </div>

                    <span className="rounded-md border border-white/[0.08] px-2.5 py-1 text-[10px] text-slate-400">
                      {report
                        ? `${report.total_samples} samples`
                        : "No results loaded"}
                    </span>
                  </div>

                  {!report ? (
                    <div className="px-5 py-12 text-center text-xs text-slate-500">
                      Run the benchmark to load live results from FastAPI.
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full min-w-[760px] text-left text-xs">
                        <thead className="bg-white/[0.025] text-slate-400">
                          <tr>
                            <th className="px-5 py-3 font-medium">Sample</th>
                            <th className="px-5 py-3 font-medium">Question</th>
                            <th className="px-5 py-3 font-medium">Expected</th>
                            <th className="px-5 py-3 font-medium">TF-IDF first</th>
                            <th className="px-5 py-3 font-medium">BM25 first</th>
                          </tr>
                        </thead>

                        <tbody>
                          {tfidf?.per_question.map((item) => {
                            const bm25Item = bm25?.per_question.find(
                              (row) => row.sample_id === item.sample_id
                            );

                            return (
                              <tr
                                key={item.sample_id}
                                className="border-t border-white/[0.06] align-top"
                              >
                                <td className="whitespace-nowrap px-5 py-4 font-mono text-cyan-200">
                                  {item.sample_id}
                                </td>

                                <td className="max-w-[270px] px-5 py-4 leading-5 text-slate-300">
                                  {item.question}
                                  <p className="mt-1 text-[10px] text-slate-500">
                                    {item.category}
                                  </p>
                                </td>

                                <td className="px-5 py-4 font-mono text-slate-400">
                                  {item.expected_source_ids.join(", ")}
                                </td>

                                <td className="px-5 py-4 font-mono text-slate-300">
                                  {item.retrieved_source_ids[0] ?? "None"}
                                </td>

                                <td className="px-5 py-4 font-mono text-slate-300">
                                  {bm25Item?.retrieved_source_ids[0] ?? "None"}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </section>
              </>
            )}

            {/* FAILURE ANALYSIS */}
            {activeView === "Failure Analysis" && (
              <>
                <section className="mb-5 flex flex-col gap-4 rounded-xl border border-white/[0.07] bg-[#101520] p-5 sm:flex-row sm:items-center sm:justify-between sm:p-6">
                  <div>
                    <h3 className="text-sm font-semibold">
                      Retrieval outcome analysis
                    </h3>
                    <p className="mt-1 max-w-2xl text-xs leading-5 text-slate-500">
                      Classifies each question as SUCCESS, RANKING_ERROR,
                      or RETRIEVAL_MISS.
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <button
                      onClick={() => void loadFailureAnalysis()}
                      disabled={failureLoading}
                      className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-cyan-300 px-5 py-3 text-xs font-semibold text-slate-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <Icon name="Failure Analysis" size={16} />
                      {failureLoading
                        ? "Analyzing..."
                        : "Load failure analysis"}
                    </button>

                    <button
                      onClick={downloadFailureAnalysis}
                      disabled={!failureAnalysis}
                      className="rounded-lg border border-white/10 bg-white/[0.04] px-4 py-3 text-xs text-slate-200 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      Download JSON
                    </button>
                  </div>
                </section>

                {!failureAnalysis ? (
                  <section className="rounded-xl border border-white/[0.07] bg-[#101520] px-5 py-12 text-center">
                    <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl border border-cyan-300/15 bg-cyan-300/[0.06] text-cyan-200">
                      <Icon name="Failure Analysis" size={22} />
                    </div>

                    <h3 className="mt-4 text-sm font-semibold">
                      Failure analysis not loaded yet
                    </h3>

                    <p className="mx-auto mt-2 max-w-md text-xs leading-6 text-slate-500">
                      Click Load failure analysis to retrieve results from FastAPI.
                    </p>
                  </section>
                ) : (
                  <>
                    <div className="mb-5 grid gap-5 xl:grid-cols-2">
                      {(["TF-IDF", "BM25"] as const).map((name) => {
                        const data = failureAnalysis.retrievers[name];
                        const accent =
                          name === "TF-IDF"
                            ? "text-cyan-200"
                            : "text-violet-200";
                        const bar =
                          name === "TF-IDF"
                            ? "bg-cyan-300"
                            : "bg-violet-300";

                        return (
                          <section
                            key={name}
                            className="rounded-xl border border-white/[0.07] bg-[#101520] p-5 sm:p-6"
                          >
                            <div className="flex items-start justify-between">
                              <div>
                                <h3 className="text-sm font-semibold">
                                  {name}
                                </h3>
                                <p className="mt-1 text-xs text-slate-500">
                                  Top-1 retrieval outcomes
                                </p>
                              </div>

                              <span className={`text-xl font-semibold ${accent}`}>
                                {(data.top1_success_rate * 100).toFixed(1)}%
                              </span>
                            </div>

                            <div className="mt-5 h-2 overflow-hidden rounded-full bg-white/[0.06]">
                              <div
                                className={`h-full rounded-full ${bar}`}
                                style={{
                                  width: `${Math.min(
                                    100,
                                    data.top1_success_rate * 100
                                  )}%`,
                                }}
                              />
                            </div>

                            <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
                              {[
                                { label: "Questions", value: data.total_questions },
                                { label: "Successes", value: data.successes },
                                { label: "Ranking errors", value: data.ranking_errors },
                                { label: "Retrieval misses", value: data.retrieval_misses },
                              ].map((stat) => (
                                <div
                                  key={stat.label}
                                  className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3"
                                >
                                  <p className="text-[10px] text-slate-500">
                                    {stat.label}
                                  </p>
                                  <p className="mt-2 text-xl font-semibold">
                                    {stat.value}
                                  </p>
                                </div>
                              ))}
                            </div>
                          </section>
                        );
                      })}
                    </div>

                    <section className="mb-5 overflow-hidden rounded-xl border border-white/[0.07] bg-[#101520]">
                      <div className="border-b border-white/[0.06] px-5 py-5 sm:px-6">
                        <h3 className="text-sm font-semibold">
                          Category breakdown
                        </h3>
                        <p className="mt-1 text-xs text-slate-500">
                          Counts provided by the backend failure analyzer.
                        </p>
                      </div>

                      <div className="overflow-x-auto">
                        <table className="w-full min-w-[570px] text-left text-xs">
                          <thead className="bg-white/[0.025] text-slate-400">
                            <tr>
                              <th className="px-5 py-3 font-medium">Retriever</th>
                              <th className="px-5 py-3 font-medium">Category</th>
                              <th className="px-5 py-3 font-medium">Outcome</th>
                              <th className="px-5 py-3 font-medium">Count</th>
                            </tr>
                          </thead>

                          <tbody>
                            {(["TF-IDF", "BM25"] as const).flatMap((name) =>
                              failureAnalysis.retrievers[name].category_summary.map(
                                (item, index) => (
                                  <tr
                                    key={`${name}-${item.category}-${item.status}-${index}`}
                                    className="border-t border-white/[0.06]"
                                  >
                                    <td className="px-5 py-3 font-medium text-slate-300">
                                      {name}
                                    </td>
                                    <td className="px-5 py-3 text-slate-300">
                                      {item.category}
                                    </td>
                                    <td className="px-5 py-3">
                                      <StatusPill status={item.status} />
                                    </td>
                                    <td className="px-5 py-3 font-mono text-slate-300">
                                      {item.count}
                                    </td>
                                  </tr>
                                )
                              )
                            )}
                          </tbody>
                        </table>
                      </div>
                    </section>

                    <section className="overflow-hidden rounded-xl border border-white/[0.07] bg-[#101520]">
                      <div className="border-b border-white/[0.06] px-5 py-5 sm:px-6">
                        <h3 className="text-sm font-semibold">
                          Failed questions
                        </h3>
                        <p className="mt-1 text-xs text-slate-500">
                          Questions classified as ranking errors or retrieval misses.
                        </p>
                      </div>

                      {(["TF-IDF", "BM25"] as const).map((name) => {
                        const failures =
                          failureAnalysis.retrievers[name].failures;

                        return (
                          <div
                            key={name}
                            className="border-b border-white/[0.06] last:border-0"
                          >
                            <div className="px-5 pt-4 text-xs font-semibold text-slate-300 sm:px-6">
                              {name}
                            </div>

                            {failures.length === 0 ? (
                              <p className="px-5 py-4 text-xs leading-5 text-emerald-200 sm:px-6">
                                No top-1 failures found in this benchmark dataset.
                              </p>
                            ) : (
                              <div className="space-y-3 p-5 sm:px-6">
                                {failures.map((item) => (
                                  <div
                                    key={item.sample_id}
                                    className="rounded-lg border border-white/[0.07] p-4"
                                  >
                                    <div className="flex flex-wrap items-center justify-between gap-2">
                                      <span className="font-mono text-xs text-cyan-200">
                                        {item.sample_id}
                                      </span>
                                      <StatusPill status={item.status} />
                                    </div>

                                    <p className="mt-3 text-xs text-slate-400">
                                      Category: {item.category}
                                    </p>
                                    <p className="mt-2 break-words text-xs leading-5 text-slate-400">
                                      Expected: {item.expected_source_ids.join(", ")}
                                    </p>
                                    <p className="mt-1 break-words text-xs leading-5 text-slate-400">
                                      Retrieved: {item.retrieved_source_ids.join(", ") || "None"}
                                    </p>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </section>
                  </>
                )}
              </>
            )}

            {/* BENCHMARKS AND RETRIEVERS */}
            {(activeView === "Benchmarks" ||
              activeView === "Retrievers") && (
              <>
                <div className="mb-5 flex flex-col gap-4 rounded-xl border border-white/[0.07] bg-[#101520] p-5 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <h3 className="text-sm font-semibold">
                      {activeView === "Benchmarks"
                        ? "Benchmark execution"
                        : "Retriever comparison"}
                    </h3>
                    <p className="mt-1 text-xs leading-5 text-slate-500">
                      {activeView === "Benchmarks"
                        ? "Execute the backend evaluation and refresh the comparison."
                        : "Compare TF-IDF and BM25 retrieval implementations."}
                    </p>
                    <p className="mt-2 text-xs text-cyan-200">
                      Selected dataset:{" "}
                      {selectedDataset === "expanded"
                        ? "Expanded — 20 questions"
                        : "Default — 5 questions"}
                    </p>
                  </div>

                  <button
                    onClick={() => void runBenchmark()}
                    disabled={benchmarkLoading}
                    className="rounded-lg bg-cyan-300 px-5 py-3 text-xs font-semibold text-slate-950 hover:bg-cyan-200 disabled:opacity-60"
                  >
                    {benchmarkLoading ? "Running..." : "Run benchmark"}
                  </button>
                </div>

                {report ? (
                  <div className="grid gap-5 xl:grid-cols-2">
                    {(["TF-IDF", "BM25"] as const).map((name) => {
                      const data = report.retrievers[name];
                      const color =
                        name === "TF-IDF"
                          ? "bg-cyan-300"
                          : "bg-violet-300";

                      return (
                        <section
                          key={name}
                          className="rounded-xl border border-white/[0.07] bg-[#101520] p-5 sm:p-6"
                        >
                          <h3 className="text-sm font-semibold">{name}</h3>
                          <p className="mt-1 text-xs text-slate-500">
                            {data.per_question.length} questions evaluated
                          </p>

                          <MetricBar
                            label="Precision@K"
                            value={data.aggregate_metrics.precision_at_k}
                            color={color}
                          />
                          <MetricBar
                            label="Recall@K"
                            value={data.aggregate_metrics.recall_at_k}
                            color={color}
                          />
                          <MetricBar
                            label="MRR@K"
                            value={data.aggregate_metrics.mrr_at_k}
                            color={color}
                          />
                        </section>
                      );
                    })}
                  </div>
                ) : (
                  <section className="rounded-xl border border-white/[0.07] bg-[#101520] p-8 text-center text-xs text-slate-500">
                    Run a benchmark to load results into this section.
                  </section>
                )}
              </>
            )}

            {/* EVALUATION RUNS */}
            {activeView === "Evaluation Runs" && <RunHistory />}

            {/* REPORTS */}
            {activeView === "Reports" && (
              <div className="space-y-5">
                <section className="rounded-xl border border-white/[0.07] bg-[#101520] p-5 sm:p-6">
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <h3 className="text-sm font-semibold">
                        Benchmark JSON report
                      </h3>
                      <p className="mt-1 break-all text-xs leading-5 text-slate-500">
                        {benchmark?.report_file ??
                          "Run a benchmark to create a report."}
                      </p>
                      <p className="mt-2 text-xs text-cyan-200">
                        Selected dataset:{" "}
                        {selectedDataset === "expanded"
                          ? "Expanded — 20 questions"
                          : "Default — 5 questions"}
                      </p>
                    </div>

                    <div className="flex flex-wrap gap-2">
                      <button
                        onClick={() => void runBenchmark()}
                        disabled={benchmarkLoading}
                        className="rounded-lg border border-white/10 bg-white/[0.04] px-4 py-3 text-xs text-slate-200 hover:bg-white/[0.08] disabled:opacity-60"
                      >
                        {benchmarkLoading ? "Running..." : "Run benchmark"}
                      </button>

                      <button
                        onClick={downloadReport}
                        disabled={!benchmark}
                        className="rounded-lg bg-cyan-300 px-4 py-3 text-xs font-semibold text-slate-950 hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        Download JSON
                      </button>
                    </div>
                  </div>

                  {report && (
                    <div className="mt-6 grid gap-4 sm:grid-cols-3">
                      {[
                        { label: "Dataset", value: report.dataset },
                        { label: "Knowledge base", value: report.knowledge_base },
                        { label: "Questions", value: String(report.total_samples) },
                      ].map((item) => (
                        <div
                          key={item.label}
                          className="rounded-xl border border-white/[0.06] p-4"
                        >
                          <p className="text-xs text-slate-500">
                            {item.label}
                          </p>
                          <p className="mt-2 break-words text-sm font-semibold">
                            {item.value}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </section>

                <section className="rounded-xl border border-white/[0.07] bg-[#101520] p-5 sm:p-6">
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <h3 className="text-sm font-semibold">
                        Failure analysis JSON
                      </h3>
                      <p className="mt-1 text-xs leading-5 text-slate-500">
                        Load failure analysis first, then download its JSON report.
                      </p>
                    </div>

                    <div className="flex flex-wrap gap-2">
                      <button
                        onClick={() => void loadFailureAnalysis()}
                        disabled={failureLoading}
                        className="rounded-lg border border-white/10 bg-white/[0.04] px-4 py-3 text-xs text-slate-200 hover:bg-white/[0.08] disabled:opacity-60"
                      >
                        {failureLoading ? "Loading..." : "Load analysis"}
                      </button>

                      <button
                        onClick={downloadFailureAnalysis}
                        disabled={!failureAnalysis}
                        className="rounded-lg bg-cyan-300 px-4 py-3 text-xs font-semibold text-slate-950 hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        Download analysis JSON
                      </button>
                    </div>
                  </div>
                </section>
              </div>
            )}

            <footer className="mt-8 flex flex-col justify-between gap-2 border-t border-white/[0.06] pt-5 text-[10px] text-slate-600 sm:flex-row">
              <span>RAG Evaluation &amp; Benchmark Lab</span>
              <a
                href={`${API_BASE}/docs`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 hover:text-cyan-200"
              >
                <Icon name="server" size={13} />
                Open FastAPI documentation
              </a>
            </footer>
          </div>
        </section>
      </div>
    </main>
  );
}
