import type { QueryResponse } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function query(question: string): Promise<QueryResponse> {
  const r = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!r.ok) throw new Error(`API error ${r.status}: ${await r.text()}`);
  return r.json();
}

export async function health(): Promise<{ status: string; reason?: string }> {
  const r = await fetch(`${BASE}/health`);
  return r.json();
}
