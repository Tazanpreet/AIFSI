"""End-to-end API tests. They train a tiny model first, so they need PyTorch + FastAPI (skipped if missing)."""
import importlib
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    base = tmp_path_factory.mktemp("aifsi")
    mp = pytest.MonkeyPatch()
    mp.setenv("AIFSI_ARTIFACTS", str(base / "artifacts"))
    mp.setenv("AIFSI_DB", str(base / "api.db"))
    mp.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens here -> fast, predictable 'unavailable'

    from data import training_rows
    from train import save_artifacts, train_gru

    model, vocab = train_gru(training_rows(n_per_class=40, seed=1), epochs=4, verbose=False)
    save_artifacts(model, vocab)

    main = importlib.import_module("backend.main")
    main = importlib.reload(main)
    with TestClient(main.app) as c:
        c.main = main
        yield c
    mp.undo()


OOM = "CRITICAL [embedding-worker] CUDA out of memory: tried to allocate 6.4 GiB, gpu has 0.7 GiB free"


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and len(body["classes"]) == 9


def test_analyze_returns_a_complete_diagnosis(client):
    r = client.post("/api/analyze", json={"text": OOM})
    assert r.status_code == 200
    d = r.json()
    for key in ["id", "label", "title", "severity", "confidence", "unknown", "scores", "tokens", "entities", "root_causes", "actions", "similar"]:
        assert key in d
    assert 0 <= d["confidence"] <= 1
    assert abs(sum(s["p"] for s in d["scores"]) - 1) < 0.01
    assert d["entities"]["memory_sizes"] == ["6.4 GiB", "0.7 GiB"]


def test_gibberish_is_flagged_unknown(client):
    d = client.post("/api/analyze", json={"text": "zzqx wvut plorb gnarf blorp"}).json()
    assert d["unknown"] is True
    assert d["label"] == "unknown"
    assert d["unknown_reason"] == "unfamiliar_vocabulary"
    assert d["top_guess"]["label"] != "unknown"


def test_validation_errors(client):
    assert client.post("/api/analyze", json={"text": "ab"}).status_code == 422
    assert client.post("/api/analyze", json={}).status_code == 422
    assert client.post("/api/analyze", json={"text": "x" * 5001}).status_code == 422
    assert client.post("/api/analyze", json={"text": "!!! ??? ..."}).status_code in (200, 400)


def test_same_log_twice_finds_the_earlier_one_as_similar(client):
    text = "429 Too Many Requests from provider; retry-after 30s, rate limit exceeded unique-marker-xyz"
    first = client.post("/api/analyze", json={"text": text}).json()
    second = client.post("/api/analyze", json={"text": text}).json()
    assert second["similar"] and second["similar"][0]["id"] == first["id"]
    assert second["similar"][0]["similarity"] >= 0.99


def test_history_incident_and_reports(client):
    iid = client.post("/api/analyze", json={"text": OOM}).json()["id"]
    hist = client.get("/api/history?limit=5").json()
    assert hist[0]["id"] == iid
    inc = client.get(f"/api/incidents/{iid}").json()
    assert inc["text"] == OOM and "result" in inc

    md = client.get(f"/api/incidents/{iid}/report.md")
    assert md.status_code == 200 and "attachment" in md.headers["content-disposition"]
    assert f"# Incident report #{iid}" in md.text
    html = client.get(f"/api/incidents/{iid}/report.html")
    assert html.status_code == 200 and "<h1>" in html.text

    for path in [f"/api/incidents/999999", f"/api/incidents/999999/report.md", f"/api/incidents/999999/report.html"]:
        assert client.get(path).status_code == 404


def test_stats_reflect_saved_incidents(client):
    s = client.get("/api/stats").json()
    assert s["total"] >= 1 and s["by_label"]


def test_ai_summary_gives_503_with_a_helpful_message_when_ollama_is_off(client):
    iid = client.post("/api/analyze", json={"text": OOM}).json()["id"]
    r = client.post(f"/api/incidents/{iid}/summary")
    assert r.status_code == 503
    assert "Ollama" in r.json()["detail"]
    assert client.post("/api/incidents/999999/summary").status_code == 404


LOG = "\n".join(
    [f"2026-11-01 10:{m:02d}:00 WARN [chat-api] 429 Too Many Requests, rate limit exceeded" for m in range(0, 10)]
    + ["", "   ", "x"]  # blank / too-short lines are ignored
)


def test_analyze_log_summarizes_lines_and_saves_a_run(client):
    r = client.post("/api/analyze-log", json={"text": LOG, "name": "t.log"})
    assert r.status_code == 200
    d = r.json()
    assert d["summary"]["total"] == 10 and len(d["lines"]) == 10
    assert d["lines"][0]["ts"].startswith("2026-11-01T10:00")
    assert d["summary"]["timeline_unit"] == "minute"
    runs = client.get("/api/log-runs").json()
    assert runs[0]["name"] == "t.log" and runs[0]["n_lines"] == 10


def test_analyze_log_rejects_empty_and_caps_long_input(client):
    assert client.post("/api/analyze-log", json={"text": "  \n ab \n"}).status_code == 400
    big = "\n".join(f"2026-11-01 10:00:{i % 60:02d} timeout error number {i}" for i in range(800))
    d = client.post("/api/analyze-log", json={"text": big}).json()
    assert d["truncated"] is True and len(d["lines"]) == 500


def test_report_endpoint_404_then_served(client):
    assert client.get("/api/report").status_code == 404
    (client.main.ART / "report.json").write_text(json.dumps({"labels": [], "experiments": {}}), encoding="utf-8")
    assert client.get("/api/report").status_code == 200


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "AI System Failure Investigator" in r.text
    assert client.get("/app.js").status_code == 200


def test_clear_history(client):
    assert client.delete("/api/history").json()["deleted"] >= 1
    assert client.get("/api/history").json() == []
