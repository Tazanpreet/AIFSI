"""Build a Markdown or printable-HTML incident report (pure Python)."""
import html

DISCLAIMER = "Suggestions only. Review before changing any production system."
REASONS = {"unfamiliar_vocabulary": "most words were unfamiliar to the model", "low_confidence": "low confidence"}


def _fence(text: str) -> str:
    return text.replace("```", "'''")


def _top_evidence(result: dict, k: int = 8, min_score: float = 0.3) -> list[str]:
    toks = sorted((t for t in result.get("tokens", []) if t["score"] >= min_score), key=lambda t: -t["score"])
    seen = []
    for t in toks:
        if t["t"] not in seen:
            seen.append(t["t"])
    return seen[:k]


def _entity_lines(result: dict) -> list[tuple[str, str]]:
    return [(k.replace("_", " "), ", ".join(v)) for k, v in result.get("entities", {}).items()]


def _unknown_note(result: dict) -> str:
    guess = result.get("top_guess", {}).get("title", "n/a")
    reason = REASONS.get(result.get("unknown_reason"), "not confident")
    return f"The model could not classify this log reliably ({reason}). Closest guess: {guess}."


def build_markdown(inc: dict) -> str:
    r = inc["result"]
    out = [
        f"# Incident report #{inc['id']}: {r['title']}", "",
        "| Field | Value |", "|---|---|",
        f"| Incident ID | {inc['id']} |",
        f"| Created (UTC) | {inc['created_at']} |",
        f"| Category | {r['title']} |",
        f"| Severity | {r['severity']} |",
        f"| Model confidence | {r['confidence'] * 100:.1f}% |", "",
    ]
    if r.get("unknown"):
        out += [f"> {_unknown_note(r)}", ""]
    out += ["## Log", "", "```", _fence(inc["text"]), "```", ""]

    evidence = _top_evidence(r)
    if evidence:
        out += ["## Evidence (words that influenced the prediction most)", "", ", ".join(f"`{w}`" for w in evidence), ""]
    ents = _entity_lines(r)
    if ents:
        out += ["## Extracted details", ""] + [f"- **{k}:** {v}" for k, v in ents] + [""]
    out += ["## Likely root causes", ""] + [f"- {c}" for c in r["root_causes"]] + [""]
    out += ["## Suggested actions", ""] + [f"{i}. {a}" for i, a in enumerate(r["actions"], 1)] + [""]
    out += [f"**Prevention:** {r['prevention']}", ""]
    if r.get("similar"):
        out += ["## Similar past incidents", ""]
        out += [f"- #{s['id']} ({s['title']}, similarity {s['similarity'] * 100:.0f}%): {s['text']}" for s in r["similar"]]
        out += [""]
    if inc.get("ai_summary"):
        out += ["## AI-written summary (local model, review before use)", "", inc["ai_summary"], ""]
    out += ["---", f"_{DISCLAIMER}_", ""]
    return "\n".join(out)


CSS = """
body{font:15px/1.55 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;max-width:800px;margin:32px auto;padding:0 20px;color:#1d2433}
h1{font-size:1.5rem;margin-bottom:4px} h2{font-size:1.05rem;margin:22px 0 6px;border-bottom:1px solid #e4e7ee;padding-bottom:4px}
table{border-collapse:collapse;margin:12px 0} td,th{border:1px solid #d0d5e0;padding:5px 10px;text-align:left}
pre{background:#f3f4f8;padding:12px;border-radius:6px;white-space:pre-wrap;word-break:break-word}
.note{background:#fff4e6;border-left:4px solid #e67700;padding:8px 12px;margin:12px 0}
.muted{color:#667085;font-size:13px} button{padding:6px 12px;margin-bottom:8px}
@media print{button{display:none}}
"""


def build_html(inc: dict) -> str:
    e = html.escape
    r = inc["result"]
    p = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>Incident report #{inc['id']}</title><style>{CSS}</style></head><body>",
        "<button onclick='window.print()'>Print / Save as PDF</button>",
        f"<h1>Incident report #{inc['id']}: {e(r['title'])}</h1>",
        "<table>",
        f"<tr><th>Incident ID</th><td>{inc['id']}</td></tr>",
        f"<tr><th>Created (UTC)</th><td>{e(inc['created_at'])}</td></tr>",
        f"<tr><th>Category</th><td>{e(r['title'])}</td></tr>",
        f"<tr><th>Severity</th><td>{e(r['severity'])}</td></tr>",
        f"<tr><th>Model confidence</th><td>{r['confidence'] * 100:.1f}%</td></tr>",
        "</table>",
    ]
    if r.get("unknown"):
        p.append(f"<div class='note'>{e(_unknown_note(r))}</div>")
    p += ["<h2>Log</h2>", f"<pre>{e(inc['text'])}</pre>"]
    evidence = _top_evidence(r)
    if evidence:
        p += ["<h2>Evidence</h2>", "<p>" + ", ".join(f"<code>{e(w)}</code>" for w in evidence) + "</p>"]
    ents = _entity_lines(r)
    if ents:
        p += ["<h2>Extracted details</h2><ul>"] + [f"<li><b>{e(k)}:</b> {e(v)}</li>" for k, v in ents] + ["</ul>"]
    p += ["<h2>Likely root causes</h2><ul>"] + [f"<li>{e(c)}</li>" for c in r["root_causes"]] + ["</ul>"]
    p += ["<h2>Suggested actions</h2><ol>"] + [f"<li>{e(a)}</li>" for a in r["actions"]] + ["</ol>"]
    p.append(f"<p><b>Prevention:</b> {e(r['prevention'])}</p>")
    if r.get("similar"):
        p.append("<h2>Similar past incidents</h2><ul>")
        p += [
            f"<li>#{s['id']} ({e(s['title'])}, similarity {s['similarity'] * 100:.0f}%): {e(s['text'])}</li>"
            for s in r["similar"]
        ]
        p.append("</ul>")
    if inc.get("ai_summary"):
        p += ["<h2>AI-written summary (local model, review before use)</h2>", f"<p>{e(inc['ai_summary'])}</p>"]
    p += [f"<p class='muted'>{e(DISCLAIMER)}</p>", "</body></html>"]
    return "".join(p)
