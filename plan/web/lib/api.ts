import { ApiError, Example, Health, QueryResponse, UnitType } from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError("network", `Cannot reach the API at ${API_BASE}`, "Start it: uvicorn --factory mmrag.api.server:create_app --port 8000");
  }
  if (!res.ok) {
    let detail: any = null;
    try {
      detail = (await res.json()).detail;
    } catch {}
    if (detail?.kind === "configuration") {
      const p = detail.problems?.[0];
      throw new ApiError("configuration", p?.problem ?? "Service is not configured", p?.fix);
    }
    if (detail?.kind === "upstream") throw new ApiError("upstream", detail.message ?? "Model call failed");
    throw new ApiError("http", `Request failed (HTTP ${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const getHealth = () => request<Health>("/health");
export const getExamples = () => request<Example[]>("/examples");
export const postQuery = (question: string, typeFilter?: UnitType | null) =>
  request<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(typeFilter ? { question, type_filter: typeFilter } : { question }),
  });
export const artefactUrl = (path: string) => `${API_BASE}${path}`;
