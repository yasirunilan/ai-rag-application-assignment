import { SourceItem } from "@/lib/types";

// Visible by default: this is the trust mechanism (Story E7.2).
export default function SourcesPanel({ sources }: { sources: SourceItem[] }) {
  return (
    <aside className="panel" aria-label="Sources">
      <h2>Sources</h2>
      {sources.length === 0 ? (
        <p className="muted">Retrieved units will be listed here, with the cited one highlighted.</p>
      ) : (
        <ol className="sources">
          {sources.map((s) => (
            <li key={s.unit_id} className={s.cited ? "cited" : ""}>
              <div><span className="chipType">{s.type}</span> <strong>{s.citation}</strong> {s.cited && <span className="badge good">cited</span>}</div>
              <div className="muted">score {s.score.toFixed(3)} · <code>{s.unit_id}</code></div>
              <div className="preview">{s.preview}</div>
            </li>
          ))}
        </ol>
      )}
    </aside>
  );
}
