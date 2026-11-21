export type UnitType = "text" | "figure" | "table";

export interface Hit {
  unit: { type: UnitType; search_text: string; artefact: unknown;
          provenance: { document: string; page: number; unit_id: string } };
  score: number;
  citation: string;
}

export interface QueryResponse {
  answer: string;
  answer_type?: UnitType;
  artefact?: unknown;
  citation?: string;
  sources: Hit[];
  abstained: boolean;
  reason?: string;
  timings: Record<string, number>;
}
