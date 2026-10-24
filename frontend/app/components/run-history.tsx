
"use client";

import { useCallback, useEffect, useState } from "react";


const API_BASE = "http://127.0.0.1:8000";

type RetrieverName = "TF-IDF" | "BM25";

interface AggregateMetrics {
  precision_at_k: number;
  recall_at_k: number;
  mrr_at_k: number;
}

interface RunSummary {
  run_id: string;
  created_at: string;
  k: number;
  dataset: string;
  knowledge_base: string;
  total_samples: number;
  knowledge_base_document_count: number;
  retriever_metrics: Record<RetrieverName, AggregateMetrics>;
  failure_summary: Record<
    RetrieverName,
    {
      successes: number;
      ranking_errors: number;
      retrieval_misses: number;
      top1_success_rate: number;
    }
  >;
}

interface QuestionResult {
  sample_id: string;
  question: string;
  category: string;
  expected_source_ids: string[];
  retrieved_source_ids: string[];
  precision_at_k: number;
  recall_at_k: number;
  reciprocal_rank_at_k: number;
}

interface FullRetrieverReport {
  aggregate_metrics: AggregateMetrics;
  per_question: QuestionResult[];
}

interface RunDetails extends RunSummary {
  report: {
    evaluation_type: string;
    dataset: string;
    knowledge_base: string;
    total_samples: number;
    knowledge_base_document_count: number;
    k: number;
    retrievers: Record<RetrieverName, FullRetrieverReport>;
  };
  failure_analysis: Record<
    RetrieverName,
    {
      total_questions: number;
      successes: number;
      ranking_errors: number;
      retrieval_misses: number;
      top1_success_rate: number;
      failures: Array<{
        sample_id: string;
        category: string;
        expected_source_ids: string[];
        retrieved_source_ids: string[];
        status: string;
      }>;
    }
  >;
}

interface ApiPayload {
  status?: string;
  count?: number;
  runs?: RunSummary[];
  run?: RunDetails;
  detail?: string;
}

function formatDate(value: string): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

function formatMetric(value: number | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(3)
    : "—";
}

function RunMetric({
  label,
  value,
}: {
  label: string;
  value: number | undefined;
}) {
  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
      <p className="text-[10px] text-slate-500">{label}</p>
      <p className="mt-2 font-mono text-sm font-semibold text-slate-200">
        {formatMetric(value)}
      </p>
    </div>
  );
}

export default function RunHistory() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingDetailId, setLoadingDetailId] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<RunDetails | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadRuns = useCallback(async () => {
    setLoading(true);
    setError("");
    setNotice("");

    try {
      const response = await fetch(`${API_BASE}/api/v1/runs?limit=50`);
      const body = (await response.json().catch(() => ({}))) as ApiPayload;

      if (!response.ok) {
        throw new Error(
          body.detail || `Could not load runs (${response.status}).`
        );
      }

      setRuns(body.runs ?? []);

      setSelectedRun((current) => {
        if (!current) return null;

        const stillExists = (body.runs ?? []).some(
          (run) => run.run_id === current.run_id
        );

        return stillExists ? current : null;
      });

      setNotice(
        `Loaded ${body.count ?? body.runs?.length ?? 0} saved evaluation run(s).`
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load evaluation history."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadRuns();
  }, [loadRuns]);

  async function viewRun(runId: string) {
    if (selectedRun?.run_id === runId) {
      setSelectedRun(null);
      return;
    }

    setLoadingDetailId(runId);
    setError("");
    setNotice("");

    try {
      const response = await fetch(
        `${API_BASE}/api/v1/runs/${encodeURIComponent(runId)}`
      );

      const body = (await response.json().catch(() => ({}))) as ApiPayload;

      if (!response.ok) {
        throw new Error(
          body.detail || `Could not load run details (${response.status}).`
        );
      }

      if (!body.run) {
        throw new Error("The API did not return the requested run.");
      }

      setSelectedRun(body.run);
      setNotice("Loaded the full saved report and failure analysis.");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load run details."
      );
    } finally {
      setLoadingDetailId(null);
    }
  }

  return (
    <div className="space-y-5">
      <section className="flex flex-col gap-4 rounded-xl border border-white/[0.07] bg-[#101520] p-5 sm:flex-row sm:items-center sm:justify-between sm:p-6">
        <div>
          <h3 className="text-sm font-semibold">Saved evaluation history</h3>
          <p className="mt-2 max-w-2xl text-xs leading-5 text-slate-500">
            Benchmark runs are stored in the local SQLite database and can
            be retrieved after the API server restarts.
          </p>
          <p className="mt-2 text-[11px] text-slate-400">
            Records loaded: {runs.length}
          </p>
        </div>

        <button
          onClick={() => void loadRuns()}
          disabled={loading}
          className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-4 py-3 text-xs font-medium text-slate-200 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Loading history..." : "Refresh history"}
        </button>
      </section>

      {error && (
        <div
          role="alert"
          className="rounded-xl border border-rose-400/20 bg-rose-400/[0.06] p-4 text-sm text-rose-200"
        >
          <p className="font-semibold">Could not load evaluation history</p>
          <p className="mt-2 break-words leading-6">{error}</p>
          <p className="mt-2 text-xs text-rose-200/70">
            Confirm that FastAPI is running at 127.0.0.1:8000.
          </p>
        </div>
      )}

      {notice && !error && (
        <div
          role="status"
          className="rounded-xl border border-emerald-300/20 bg-emerald-300/[0.06] p-3 text-xs text-emerald-200"
        >
          {notice}
        </div>
      )}

      {!loading && !error && runs.length === 0 && (
        <section className="rounded-xl border border-white/[0.07] bg-[#101520] px-5 py-12 text-center">
          <h3 className="text-sm font-semibold">No saved runs yet</h3>
          <p className="mx-auto mt-2 max-w-md text-xs leading-6 text-slate-500">
            Go to Overview and run a benchmark. Completed runs will appear here
            after the backend saves them.
          </p>
        </section>
      )}

      <div className="space-y-4">
        {runs.map((run) => {
          const expanded = selectedRun?.run_id === run.run_id;

          return (
            <section
              key={run.run_id}
              className="overflow-hidden rounded-xl border border-white/[0.07] bg-[#101520]"
            >
              <div className="p-5 sm:p-6">
                <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-start">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="rounded-md border border-emerald-300/15 bg-emerald-300/[0.07] px-2 py-1 text-[10px] font-medium text-emerald-200">
                        Completed
                      </span>
                      <span className="text-[10px] text-slate-500">
                        Top K = {run.k}
                      </span>
                    </div>

                    <h3 className="mt-3 text-sm font-semibold">
                      {run.dataset} · {run.knowledge_base}
                    </h3>

                    <p className="mt-2 text-xs text-slate-400">
                      {formatDate(run.created_at)}
                    </p>

                    <p className="mt-2 break-all font-mono text-[10px] leading-5 text-slate-600">
                      Run ID: {run.run_id}
                    </p>
                  </div>

                  <button
                    onClick={() => void viewRun(run.run_id)}
                    disabled={loadingDetailId === run.run_id}
                    className="shrink-0 rounded-lg bg-cyan-300 px-4 py-3 text-xs font-semibold text-slate-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {loadingDetailId === run.run_id
                      ? "Loading details..."
                      : expanded
                        ? "Hide details"
                        : "View details"}
                  </button>
                </div>

                <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3">
                  <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
                    <p className="text-[10px] text-slate-500">Questions</p>
                    <p className="mt-2 text-xl font-semibold">
                      {run.total_samples}
                    </p>
                  </div>
                  <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
                    <p className="text-[10px] text-slate-500">Documents</p>
                    <p className="mt-2 text-xl font-semibold">
                      {run.knowledge_base_document_count}
                    </p>
                  </div>
                  <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
                    <p className="text-[10px] text-slate-500">K value</p>
                    <p className="mt-2 text-xl font-semibold">{run.k}</p>
                  </div>
                </div>

                <div className="mt-5 grid gap-4 xl:grid-cols-2">
                  {(["TF-IDF", "BM25"] as const).map((name) => {
                    const metrics = run.retriever_metrics[name];

                    return (
                      <div
                        key={name}
                        className="rounded-xl border border-white/[0.06] p-4"
                      >
                        <h4 className="text-xs font-semibold">{name}</h4>
                        <div className="mt-3 grid grid-cols-3 gap-2">
                          <RunMetric
                            label="Precision@K"
                            value={metrics?.precision_at_k}
                          />
                          <RunMetric
                            label="Recall@K"
                            value={metrics?.recall_at_k}
                          />
                          <RunMetric
                            label="MRR@K"
                            value={metrics?.mrr_at_k}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {expanded && selectedRun && (
                <div className="border-t border-white/[0.07] bg-black/10 p-5 sm:p-6">
                  <h4 className="text-sm font-semibold">
                    Detailed run information
                  </h4>

                  <div className="mt-4 grid gap-4 sm:grid-cols-2">
                    {(["TF-IDF", "BM25"] as const).map((name) => {
                      const failure = selectedRun.failure_analysis[name];

                      return (
                        <div
                          key={name}
                          className="rounded-xl border border-white/[0.06] p-4"
                        >
                          <h5 className="text-xs font-semibold">{name} · Failure analysis</h5>
                          <div className="mt-4 grid grid-cols-2 gap-3">
                            {[
                              { label: "Top-1 successes", value: failure.successes },
                              { label: "Success rate", value: `${(failure.top1_success_rate * 100).toFixed(1)}%` },
                              { label: "Ranking errors", value: failure.ranking_errors },
                              { label: "Retrieval misses", value: failure.retrieval_misses },
                            ].map((metric) => (
                              <div key={metric.label}>
                                <p className="text-[10px] text-slate-500">
                                  {metric.label}
                                </p>
                                <p className="mt-1 text-sm font-semibold text-slate-200">
                                  {metric.value}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="mt-6">
                    <h4 className="text-xs font-semibold">
                      Per-question metrics
                    </h4>

                    <div className="mt-3 overflow-x-auto rounded-lg border border-white/[0.06]">
                      <table className="w-full min-w-[700px] text-left text-xs">
                        <thead className="bg-white/[0.025] text-slate-400">
                          <tr>
                            <th className="px-4 py-3 font-medium">Sample</th>
                            <th className="px-4 py-3 font-medium">Retriever</th>
                            <th className="px-4 py-3 font-medium">Precision@K</th>
                            <th className="px-4 py-3 font-medium">Recall@K</th>
                            <th className="px-4 py-3 font-medium">MRR@K</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(["TF-IDF", "BM25"] as const).flatMap((name) =>
                            selectedRun.report.retrievers[name].per_question.map(
                              (question) => (
                                <tr
                                  key={`${name}-${question.sample_id}`}
                                  className="border-t border-white/[0.06]"
                                >
                                  <td className="px-4 py-3 font-mono text-cyan-200">
                                    {question.sample_id}
                                  </td>
                                  <td className="px-4 py-3 text-slate-300">
                                    {name}
                                  </td>
                                  <td className="px-4 py-3 font-mono text-slate-300">
                                    {formatMetric(question.precision_at_k)}
                                  </td>
                                  <td className="px-4 py-3 font-mono text-slate-300">
                                    {formatMetric(question.recall_at_k)}
                                  </td>
                                  <td className="px-4 py-3 font-mono text-slate-300">
                                    {formatMetric(question.reciprocal_rank_at_k)}
                                  </td>
                                </tr>
                              )
                            )
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}
            </section>
          );
        })}
      </div>

      <p className="text-[10px] leading-5 text-slate-600">
        History is currently stored in the local SQLite database. For online
        deployment, configure persistent database storage before relying on
        the history across server instances or redeployments.
      </p>
    </div>
  );
}
