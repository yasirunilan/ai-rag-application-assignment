"""API service tests (Story E5.3): same entry points for the UI and the evaluator."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mmrag.api.server import create_app
from mmrag.config import AppConfig
from mmrag.pipeline import MultimodalRAGPipeline


@pytest.fixture()
def client(lab_cfg: AppConfig) -> TestClient:
    MultimodalRAGPipeline(lab_cfg, force_offline=True).ingest_and_index(persist_index=True)
    return TestClient(create_app(lab_cfg, offline=True))


def test_health_and_examples(client: TestClient) -> None:
    h = client.get("/health").json()
    assert h["status"] == "healthy" and set(h["units"]) == {"text", "figure", "table"}
    ex = client.get("/examples").json()
    assert len(ex) >= 3 and {"figure", "table"} <= {e["type"] for e in ex}


def test_figure_question_returns_real_image_with_citation_and_sources(client: TestClient) -> None:
    r = client.post("/query", json={"question": "Show the engine starter relay control wiring diagram for Model 737-800"})
    body = r.json()
    assert r.status_code == 200 and not body["abstained"]
    assert body["modality"] == "figure" and body["citation"].startswith("amt_handbook p.")
    assert any(s["cited"] for s in body["sources"]) and body["timings_ms"]["total"] >= 0
    img = client.get(body["image_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert img.content[:4] == b"\x89PNG"


def test_table_question_returns_exact_structured_rows(client: TestClient) -> None:
    body = client.post(
        "/query", json={"question": "What is the dry and lubed torque spec table for bolt part number AN4-12A?"}
    ).json()
    assert body["modality"] == "table" and isinstance(body["table"]["rows"], list)
    assert any("AN4-12A" in cell for row in body["table"]["rows"] for cell in row)
    assert body["citation_path"] == "amt_handbook > p.1 > amt_handbook__p0001__table_001"
    assert len(body["matched_rows"]) == 1
    assert "AN4-12A" in body["table"]["rows"][body["matched_rows"][0]]
    assert "AN4-12A" in (body["grounded_answer"] or "")
    tbl_art = client.get(f"/artefact/{body['unit_id']}")
    assert tbl_art.status_code == 200 and "headers" in tbl_art.json()


def test_text_question_and_off_topic_abstention_via_api(client: TestClient) -> None:
    txt = client.post(
        "/query",
        json={"question": "What nitrogen pre-charge pressure is required on the Model A320 hydraulic accumulator at 20 degrees Celsius?"},
    ).json()
    assert txt["modality"] == "text" and "1,450 PSI" in (txt["grounded_answer"] or "")
    txt_art = client.get(f"/artefact/{txt['unit_id']}")
    assert txt_art.status_code == 200 and "1,450 PSI" in txt_art.json()["text"]

    off = client.post("/query", json={"question": "How do I bake a chocolate cake?"}).json()
    assert off["abstained"] is True and off["abstain_code"] == "no_relevant_hits"


def test_unknown_or_unsourced_artefact_is_404_and_traversal_is_blocked(client: TestClient) -> None:
    assert client.get("/artefact/does-not-exist").status_code == 404
    assert client.get("/artefact/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_invalid_request_is_rejected(client: TestClient) -> None:
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "x", "type_filter": "audio"}).status_code == 422


def test_missing_index_is_a_configuration_error_not_a_crash(lab_cfg: AppConfig) -> None:
    c = TestClient(create_app(lab_cfg, offline=True))  # nothing ingested yet
    assert c.get("/health").json()["status"] == "degraded"
    r = c.post("/query", json={"question": "anything"})
    assert r.status_code == 503 and r.json()["detail"]["kind"] == "configuration"


def test_upstream_failure_is_a_distinct_502(client: TestClient, monkeypatch) -> None:
    svc = client.app.state.svc

    def boom(*a, **k):
        raise TimeoutError("embedding endpoint timed out")

    monkeypatch.setattr(svc.retriever, "retrieve", boom)
    r = client.post("/query", json={"question": "anything"})
    assert r.status_code == 502 and r.json()["detail"]["kind"] == "upstream"
