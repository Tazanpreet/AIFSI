"""FastAPI backend. Run from the project root:  uvicorn backend.main:app --reload"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend"))

import incident_report  # noqa: E402
import loganalysis  # noqa: E402
import ollama_client  # noqa: E402
import storage  # noqa: E402
from common import MAX_LEN, UNK, FailureClassifier, encode  # noqa: E402
from nlp_utils import KB, extract_entities  # noqa: E402
from paths import artifacts_dir  # noqa: E402
from textproc import normalize, tokenize  # noqa: E402

ART = artifacts_dir()
if not (ART / "model.pt").exists():
    raise SystemExit("Model not found. Train first:  python ml/train.py")

ckpt = torch.load(ART / "model.pt", map_location="cpu", weights_only=True)
vocab = json.loads((ART / "vocab.json").read_text(encoding="utf-8"))
LABELS = ckpt["labels"]
model = FailureClassifier(ckpt["vocab_size"], len(LABELS))
model.load_state_dict(ckpt["state_dict"])
model.eval()
MODEL_ID = str(int((ART / "model.pt").stat().st_mtime))  # changes whenever you retrain

# A log is "unknown" when the model is unsure OR most of its words were never seen in training.
UNKNOWN_THRESHOLD = float(os.environ.get("AIFSI_UNKNOWN_THRESHOLD", "0.5"))
MIN_COVERAGE = float(os.environ.get("AIFSI_MIN_COVERAGE", "0.5"))
SIM_THRESHOLD = float(os.environ.get("AIFSI_SIMILARITY_THRESHOLD", "0.85"))

storage.init()

app = FastAPI(title="AI System Failure Investigator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AnalyzeIn(BaseModel):
    text: str = Field(min_length=3, max_length=5000)


class LogIn(BaseModel):
    text: str = Field(min_length=3, max_length=500_000)
    name: str | None = Field(default=None, max_length=200)


# ---------------------------------------------------------------- model helpers
def coverage(tokens: list[str]) -> float:
    """Share of the alphabetic words that the model saw during training."""
    words = [normalize(t) for t in tokens if t[:1].isalpha()]
    if not words:
        return 0.0
    return sum(1 for w in words if w in vocab) / len(words)


@torch.no_grad()
def classify(texts: list[str]) -> list[dict]:
    out = []
    token_lists = [tokenize(t)[:MAX_LEN] for t in texts]
    for i in range(0, len(token_lists), 64):
        chunk = token_lists[i:i + 64]
        probs = torch.softmax(model(torch.tensor([encode(t, vocab) for t in chunk])), dim=1)
        for toks, p in zip(chunk, probs):
            top = int(p.argmax())
            conf = float(p[top])
            cov = coverage(toks)
            reason = "unfamiliar_vocabulary" if cov < MIN_COVERAGE else "low_confidence" if conf < UNKNOWN_THRESHOLD else None
            out.append({"top": LABELS[top], "top_idx": top, "confidence": conf, "coverage": cov,
                        "unknown": reason is not None, "unknown_reason": reason, "p": p})
    return out


@torch.no_grad()
def embed_text(tokens: list[str]) -> torch.Tensor:
    vec = model.embed(torch.tensor([encode(tokens, vocab)]))[0]
    return F.normalize(vec, dim=0)


def find_similar(vec: torch.Tensor, limit: int = 3) -> list[dict]:
    rows = storage.recent_embeddings(MODEL_ID, 500)
    if not rows:
        return []
    sims = torch.tensor([r["embedding"] for r in rows]) @ vec
    out = []
    for i in torch.argsort(sims, descending=True)[:limit].tolist():
        if float(sims[i]) >= SIM_THRESHOLD:
            r = rows[i]
            out.append({"id": r["id"], "similarity": round(float(sims[i]), 3), "text": r["text"][:160],
                        "title": r["title"], "created_at": r["created_at"]})
    return out


# ---------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"status": "ok", "classes": LABELS, "model_id": MODEL_ID, "unknown_threshold": UNKNOWN_THRESHOLD}


@app.post("/api/analyze")
def analyze(body: AnalyzeIn):
    text = body.text.strip()
    tokens = tokenize(text)[:MAX_LEN]  # the rest is ignored by the model
    if not tokens:
        raise HTTPException(400, "No analyzable text")

    d = classify([text])[0]
    label = "unknown" if d["unknown"] else d["top"]
    info = KB[label]

    # Explainability by occlusion: how much does the top-class probability drop
    # when each token is replaced with <unk>?
    with torch.no_grad():
        variants = torch.tensor([encode(tokens[:i] + [UNK] + tokens[i + 1:], vocab) for i in range(len(tokens))])
        drops = (d["p"][d["top_idx"]] - torch.softmax(model(variants), dim=1)[:, d["top_idx"]]).tolist()
    mx = max(max(drops), 1e-6)
    scored = [{"t": t, "score": round(max(x, 0) / mx, 3)} for t, x in zip(tokens, drops)]

    vec = embed_text(tokens)
    similar = find_similar(vec)
    ranked = sorted(zip(LABELS, d["p"].tolist()), key=lambda p: -p[1])

    result = {
        "label": label,
        "title": info["title"],
        "severity": info["severity"],
        "confidence": round(d["confidence"], 4),
        "unknown": d["unknown"],
        "unknown_reason": d["unknown_reason"],
        "low_confidence": d["confidence"] < 0.7,
        "top_guess": {"label": d["top"], "title": KB[d["top"]]["title"]},
        "scores": [{"label": l, "p": round(p, 4)} for l, p in ranked],
        "tokens": scored,
        "entities": extract_entities(text),
        "root_causes": info["root_causes"],
        "actions": info["actions"],
        "prevention": info["prevention"],
        "similar": similar,
        "note": "Suggestions only. Review before changing any production system.",
    }
    incident_id = storage.add_incident(text, result, embedding=[round(x, 4) for x in vec.tolist()], model_id=MODEL_ID)
    return {"id": incident_id, **result}


@app.post("/api/analyze-log")
def analyze_log(body: LogIn):
    all_lines = loganalysis.split_lines(body.text, max_lines=10**9)
    if not all_lines:
        raise HTTPException(400, "No readable lines found")
    lines = all_lines[:loganalysis.MAX_LINES]

    items = []
    for n, (line, d) in enumerate(zip(lines, classify(lines)), start=1):
        label = "unknown" if d["unknown"] else d["top"]
        ts = loganalysis.parse_ts(line)
        items.append({
            "line_no": n, "text": line[:300], "label": label, "title": KB[label]["title"],
            "severity": KB[label]["severity"], "confidence": round(d["confidence"], 3),
            "unknown": d["unknown"], "ts": ts.isoformat() if ts else None,
        })
    summary = loganalysis.summarize(items)
    name = body.name or "pasted log"
    run_id = storage.add_log_run(name, len(items), summary)
    return {"run_id": run_id, "name": name, "truncated": len(all_lines) > len(lines), "summary": summary, "lines": items}


@app.get("/api/history")
def history(limit: int = Query(20, ge=1, le=100)):
    return storage.list_incidents(limit)


@app.delete("/api/history")
def clear_history():
    return {"deleted": storage.clear_incidents()}


@app.get("/api/stats")
def stats():
    return storage.stats()


@app.get("/api/log-runs")
def log_runs(limit: int = Query(10, ge=1, le=50)):
    return storage.list_log_runs(limit)


def _incident_or_404(incident_id: int) -> dict:
    inc = storage.get_incident(incident_id)
    if inc is None:
        raise HTTPException(404, "Incident not found")
    return inc


@app.get("/api/incidents/{incident_id}")
def get_incident(incident_id: int):
    return _incident_or_404(incident_id)


@app.get("/api/incidents/{incident_id}/report.md")
def report_markdown(incident_id: int):
    inc = _incident_or_404(incident_id)
    return PlainTextResponse(
        incident_report.build_markdown(inc),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="incident-{incident_id}.md"'},
    )


@app.get("/api/incidents/{incident_id}/report.html", response_class=HTMLResponse)
def report_html(incident_id: int):
    return HTMLResponse(incident_report.build_html(_incident_or_404(incident_id)))


@app.post("/api/incidents/{incident_id}/summary")
def ai_summary(incident_id: int):
    inc = _incident_or_404(incident_id)
    try:
        text = ollama_client.summarize(inc["text"], inc["result"])
    except ollama_client.LLMUnavailable as e:
        raise HTTPException(503, str(e))
    storage.set_summary(incident_id, text)
    return {"summary": text}


@app.get("/api/report")
def model_report():
    path = ART / "report.json"
    if not path.exists():
        raise HTTPException(404, "No evaluation report yet. Run:  python ml/evaluate.py")
    return json.loads(path.read_text(encoding="utf-8"))


# Serve the frontend from the same server (avoids CORS headaches)
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
