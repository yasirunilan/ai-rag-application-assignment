import { Health } from "@/lib/types";

export default function StatusBadge({ health, error, lastMs }: { health: Health | null; error: string | null; lastMs: number | null }) {
  const ok = health?.status === "healthy";
  const label = error ? "Unreachable" : !health ? "Checking…" : ok ? "Healthy" : "Degraded";
  const cls = error || (health && !ok) ? "badge bad" : ok ? "badge good" : "badge";
  return (
    <div className="statusRow" role="status">
      <span className={cls}>{label}</span>
      {health && ok && (
        <span className="muted">
          {Object.entries(health.units).map(([t, n]) => `${n} ${t}`).join(" · ")} · {health.embedding_model}
          {health.offline ? " · offline" : ""}
        </span>
      )}
      {error && <span className="muted">{error}</span>}
      {health && !ok && health.problems.map((p) => <span key={p.problem} className="muted">{p.problem} — fix: {p.fix}</span>)}
      {lastMs !== null && <span className="muted">last answer {Math.round(lastMs)} ms</span>}
    </div>
  );
}
