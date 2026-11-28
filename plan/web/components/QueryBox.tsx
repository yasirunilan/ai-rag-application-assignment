"use client";
import { useState } from "react";
import { Example, UnitType } from "@/lib/types";

const FILTER_OPTIONS: { label: string; value: UnitType | null }[] = [
  { label: "All types", value: null },
  { label: "Text", value: "text" },
  { label: "Figure", value: "figure" },
  { label: "Table", value: "table" },
];

export default function QueryBox({
  examples,
  busy,
  onAsk,
}: {
  examples: Example[];
  busy: boolean;
  onAsk: (q: string, typeFilter: UnitType | null) => void;
}) {
  const [q, setQ] = useState("");
  const [typeFilter, setTypeFilter] = useState<UnitType | null>(null);

  const ask = (text: string, filterOverride?: UnitType | null) => {
    const activeFilter = filterOverride !== undefined ? filterOverride : typeFilter;
    setQ(text);
    if (text.trim()) onAsk(text.trim(), activeFilter);
  };

  return (
    <section>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(q);
        }}
        className="queryForm"
      >
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask a maintenance question, e.g. “show the relay control wiring”"
          aria-label="Question"
          disabled={busy}
        />
        <select
          className="typeSelect"
          aria-label="Content type filter"
          value={typeFilter ?? ""}
          disabled={busy}
          onChange={(e) => setTypeFilter((e.target.value || null) as UnitType | null)}
        >
          {FILTER_OPTIONS.map((opt) => (
            <option key={opt.label} value={opt.value ?? ""}>
              {opt.label}
            </option>
          ))}
        </select>
        <button type="submit" disabled={busy || !q.trim()}>
          {busy ? "Searching…" : "Ask"}
        </button>
      </form>
      <div className="examples">
        {examples.map((e) => (
          <button key={e.question} className="chip" disabled={busy} onClick={() => ask(e.question, null)} title={e.question}>
            <span className="chipType">{e.type}</span> {e.question}
          </button>
        ))}
      </div>
    </section>
  );
}
