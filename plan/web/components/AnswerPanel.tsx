import { ApiError, QueryResponse } from "@/lib/types";
import { artefactUrl } from "@/lib/api";

export default function AnswerPanel({ answer, busy, error }: { answer: QueryResponse | null; busy: boolean; error: ApiError | null }) {
  if (busy) return <div className="panel" aria-busy="true"><div className="spinner" /> Retrieving…</div>;
  if (error)
    return (
      <div className="panel errorPanel" role="alert">
        <strong>{error.kind === "configuration" ? "Service not ready" : error.kind === "upstream" ? "Model problem" : "Request failed"}</strong>
        <p>{error.message}</p>
        {error.fix && <p className="muted">Fix: {error.fix}</p>}
      </div>
    );
  if (!answer) return <div className="panel muted">Ask a question or click an example. Answers show the real figure or table and its page.</div>;
  if (answer.abstained)
    return (
      <div className="panel abstain">
        <strong>No sourced answer</strong>
        <p>The system abstained rather than guess: <code>{answer.abstain_reason}</code></p>
      </div>
    );
  const matched = new Set(answer.matched_rows ?? []);
  return (
    <div className="panel">
      <div className="citation">
        Source: <strong>{answer.citation}</strong>
        {answer.citation_path ? <> · <code>{answer.citation_path}</code></> : answer.unit_id ? <> · <code>{answer.unit_id}</code></> : null}
      </div>
      {answer.caption && <p className="caption">{answer.caption}</p>}
      {answer.grounded_answer && answer.modality !== "text" && (
        <p className="groundedAnswer">{answer.grounded_answer}</p>
      )}
      {answer.modality === "figure" && answer.image_url && (
        // eslint-disable-next-line @next/next/no-img-element
        <img className="figure" src={artefactUrl(answer.image_url)} alt={answer.caption ?? "Retrieved figure"} />
      )}
      {answer.modality === "table" && answer.table && (
        <div className="tableWrap">
          <table>
            <thead><tr>{answer.table.headers.map((h, i) => <th key={i}>{h}</th>)}</tr></thead>
            <tbody>
              {answer.table.rows.map((r, i) => (
                <tr key={i} className={matched.has(i) ? "matchedRow" : ""}>
                  {r.map((c, j) => <td key={j}>{c}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {answer.modality === "text" && (answer.grounded_answer || answer.text) && (
        <blockquote>{answer.grounded_answer ?? answer.text}</blockquote>
      )}
    </div>
  );
}
