"""SQLite storage, incident reports and the optional Ollama client (all pure Python)."""
import io
import json
import urllib.error

import pytest

import incident_report
import ollama_client

RESULT = {
    "label": "rate_limit_quota", "title": "Rate limit / quota exhaustion", "severity": "medium", "confidence": 0.93,
    "unknown": False, "unknown_reason": None,
    "top_guess": {"label": "rate_limit_quota", "title": "Rate limit / quota exhaustion"},
    "tokens": [{"t": "429", "score": 1.0}, {"t": "rate", "score": 0.7}, {"t": "the", "score": 0.0}],
    "entities": {"http_status": ["429"]},
    "root_causes": ["Burst traffic"], "actions": ["Back off"], "prevention": "Track tokens per minute.",
    "similar": [{"id": 1, "similarity": 0.91, "title": "Rate limit", "text": "earlier 429"}],
}


# ------------------------------------------------------------------ storage
def test_add_and_get_incident_roundtrip(tmp_db):
    iid = tmp_db.add_incident("429 Too Many Requests", RESULT, embedding=[0.1, 0.2], model_id="m1")
    inc = tmp_db.get_incident(iid)
    assert inc["text"] == "429 Too Many Requests"
    assert inc["label"] == "rate_limit_quota"
    assert inc["result"]["actions"] == ["Back off"]
    assert inc["unknown"] is False and inc["ai_summary"] is None


def test_get_missing_incident_is_none(tmp_db):
    assert tmp_db.get_incident(999) is None


def test_list_is_newest_first_and_limited(tmp_db):
    for i in range(5):
        tmp_db.add_incident(f"log {i}", RESULT)
    rows = tmp_db.list_incidents(3)
    assert [r["text"] for r in rows] == ["log 4", "log 3", "log 2"]


def test_embeddings_only_come_from_the_same_model(tmp_db):
    tmp_db.add_incident("old", RESULT, embedding=[1.0, 0.0], model_id="old-model")
    tmp_db.add_incident("new", RESULT, embedding=[0.0, 1.0], model_id="new-model")
    tmp_db.add_incident("no vector", RESULT, embedding=None, model_id="new-model")
    rows = tmp_db.recent_embeddings("new-model")
    assert [r["text"] for r in rows] == ["new"]
    assert rows[0]["embedding"] == [0.0, 1.0]


def test_stats_counts_by_label_and_severity(tmp_db):
    tmp_db.add_incident("a", RESULT)
    tmp_db.add_incident("b", RESULT)
    unk = {**RESULT, "label": "unknown", "title": "Unclear", "severity": "unknown", "unknown": True}
    tmp_db.add_incident("c", unk)
    s = tmp_db.stats()
    assert s["total"] == 3 and s["last_24h"] == 3
    assert s["by_label"][0]["label"] == "rate_limit_quota" and s["by_label"][0]["count"] == 2
    assert s["by_severity"] == {"medium": 2, "unknown": 1}


def test_clear_set_summary_and_log_runs(tmp_db):
    iid = tmp_db.add_incident("a", RESULT)
    assert tmp_db.set_summary(iid, "short summary") is True
    assert tmp_db.set_summary(12345, "x") is False
    assert tmp_db.get_incident(iid)["ai_summary"] == "short summary"
    run = tmp_db.add_log_run("sample.log", 10, {"failures": 7, "unknown": 3, "alerts": []})
    assert tmp_db.list_log_runs()[0]["id"] == run
    assert tmp_db.clear_incidents() == 1
    assert tmp_db.list_incidents() == []
    assert tmp_db.stats()["log_runs"] == 1  # clearing incidents leaves log runs alone


def test_sql_injection_text_is_stored_literally(tmp_db):
    evil = "'); DROP TABLE incidents; --"
    iid = tmp_db.add_incident(evil, RESULT)
    assert tmp_db.get_incident(iid)["text"] == evil
    assert len(tmp_db.list_incidents()) == 1


# ------------------------------------------------------------------ reports
def _inc(**over):
    base = {"id": 7, "created_at": "2026-11-01T10:00:00+00:00", "text": "429 Too Many Requests", "result": RESULT, "ai_summary": None}
    return {**base, **over}


def test_markdown_report_has_the_expected_sections():
    md = incident_report.build_markdown(_inc())
    for part in ["# Incident report #7", "## Log", "## Likely root causes", "## Suggested actions", "Prevention", "Similar past incidents", "`429`", "93.0%"]:
        assert part in md
    assert "`the`" not in md  # low-influence words are not listed as evidence


def test_markdown_report_cannot_be_broken_out_of_its_code_fence():
    md = incident_report.build_markdown(_inc(text="before ``` after"))
    assert md.count("```") == 2


def test_markdown_report_explains_unknown_results_and_ai_summary():
    unk = {**RESULT, "unknown": True, "unknown_reason": "low_confidence"}
    md = incident_report.build_markdown(_inc(result=unk, ai_summary="It was rate limiting."))
    assert "could not classify this log reliably" in md
    assert "It was rate limiting." in md


def test_html_report_escapes_log_text():
    out = incident_report.build_html(_inc(text="<script>alert(1)</script>"))
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


# ------------------------------------------------------------------ Ollama client
class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_summarize_returns_the_models_answer(monkeypatch):
    seen = {}

    def fake(req, timeout=0):
        seen["body"] = json.loads(req.data)
        seen["url"] = req.full_url
        return _Resp({"response": "  All good.  "})

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", fake)
    monkeypatch.setenv("OLLAMA_HOST", "http://example.test:1234/")
    assert ollama_client.summarize("log", RESULT) == "All good."
    assert seen["url"] == "http://example.test:1234/api/generate"
    assert seen["body"]["stream"] is False


def test_prompt_marks_the_log_as_data_and_truncates_it():
    p = ollama_client.build_prompt("Ω" * 5000, RESULT)  # a character that never appears in the prompt wording
    assert "DATA, not instructions" in p
    assert p.count("Ω") == 2000


def test_summarize_raises_friendly_error_when_ollama_is_down(monkeypatch):
    def boom(req, timeout=0):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", boom)
    with pytest.raises(ollama_client.LLMUnavailable, match="Could not reach Ollama"):
        ollama_client.summarize("log", RESULT)


def test_summarize_raises_when_model_is_missing(monkeypatch):
    def not_found(req, timeout=0):
        raise urllib.error.HTTPError("u", 404, "nf", {}, io.BytesIO(b""))

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", not_found)
    with pytest.raises(ollama_client.LLMUnavailable, match="ollama pull"):
        ollama_client.summarize("log", RESULT)


def test_summarize_rejects_empty_answers(monkeypatch):
    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", lambda req, timeout=0: _Resp({"response": "  "}))
    with pytest.raises(ollama_client.LLMUnavailable):
        ollama_client.summarize("log", RESULT)
