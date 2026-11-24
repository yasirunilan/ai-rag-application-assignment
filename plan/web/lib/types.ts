// Mirrors QueryResponse in src/mmrag/api/server.py (Story E5.3).
export type UnitType = "text" | "figure" | "table";

export interface SourceItem {
  rank: number;
  unit_id: string;
  type: UnitType;
  citation: string; // "doc_id p.N"
  score: number;
  cited: boolean;
  preview: string;
}

export interface TableData {
  headers: string[];
  rows: string[][];
  caption?: string | null;
}

export interface QueryResponse {
  question: string;
  abstained: boolean;
  abstain_reason?: string | null;
  modality?: UnitType | null;
  citation?: string | null;
  unit_id?: string | null;
  caption?: string | null;
  image_url?: string | null;
  table?: TableData | null;
  text?: string | null;
  sources: SourceItem[];
  timings_ms: Record<string, number>;
}

export interface Health {
  status: "healthy" | "degraded";
  problems: { problem: string; fix: string }[];
  units: Record<string, number>;
  embedding_model: string;
  offline: boolean;
}

export interface Example {
  type: UnitType;
  question: string;
}

/** Failure the UI can explain: configuration vs upstream vs network. */
export class ApiError extends Error {
  constructor(public kind: "configuration" | "upstream" | "network" | "http", message: string, public fix?: string) {
    super(message);
  }
}
