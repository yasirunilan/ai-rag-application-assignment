"use client";
import { useEffect, useState } from "react";
import AnswerPanel from "@/components/AnswerPanel";
import QueryBox from "@/components/QueryBox";
import SourcesPanel from "@/components/SourcesPanel";
import StatusBadge from "@/components/StatusBadge";
import { getExamples, getHealth, postQuery } from "@/lib/api";
import { ApiError, Example, Health, QueryResponse } from "@/lib/types";

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [answer, setAnswer] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);

  const refreshHealth = () =>
    getHealth().then((h) => { setHealth(h); setHealthError(null); }).catch((e: ApiError) => { setHealth(null); setHealthError(`${e.message}${e.fix ? ` — ${e.fix}` : ""}`); });

  useEffect(() => {
    refreshHealth();
    getExamples().then(setExamples).catch(() => setExamples([]));
  }, []);

  async function ask(q: string) {
    setBusy(true); setError(null);
    try {
      setAnswer(await postQuery(q));
    } catch (e) {
      setAnswer(null);
      setError(e instanceof ApiError ? e : new ApiError("http", String(e)));
      refreshHealth();
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <header><h1>Manual Finder</h1><p className="muted">Retrieve by text, answer with the real figure or table.</p></header>
      <StatusBadge health={health} error={healthError} lastMs={answer?.timings_ms?.total ?? null} />
      <QueryBox examples={examples} busy={busy} onAsk={ask} />
      <div className="grid">
        <AnswerPanel answer={answer} busy={busy} error={error} />
        <SourcesPanel sources={answer?.sources ?? []} />
      </div>
    </main>
  );
}
