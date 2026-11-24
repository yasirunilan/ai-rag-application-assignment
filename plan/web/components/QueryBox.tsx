"use client";
import { useState } from "react";
import { Example } from "@/lib/types";

export default function QueryBox({ examples, busy, onAsk }: { examples: Example[]; busy: boolean; onAsk: (q: string) => void }) {
  const [q, setQ] = useState("");
  const ask = (text: string) => {
    setQ(text);
    if (text.trim()) onAsk(text.trim());
  };
  return (
    <section>
      <form onSubmit={(e) => { e.preventDefault(); ask(q); }} className="queryForm">
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask a maintenance question, e.g. “show the relay control wiring”" aria-label="Question" disabled={busy} />
        <button type="submit" disabled={busy || !q.trim()}>{busy ? "Searching…" : "Ask"}</button>
      </form>
      <div className="examples">
        {examples.map((e) => (
          <button key={e.question} className="chip" disabled={busy} onClick={() => ask(e.question)} title={e.question}>
            <span className="chipType">{e.type}</span> {e.question}
          </button>
        ))}
      </div>
    </section>
  );
}
