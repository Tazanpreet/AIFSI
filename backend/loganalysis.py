"""Multi-line log analysis helpers (pure Python): splitting, timeline buckets, trend alerts."""
import math
import re
from collections import Counter
from datetime import datetime, timedelta

TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")
MAX_LINES = 500
MAX_LINE_CHARS = 1000
MAX_BUCKETS = 200


def parse_ts(line: str):
    m = TS_RE.search(line)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def split_lines(text: str, max_lines: int = MAX_LINES) -> list[str]:
    """Non-empty lines (at least 3 characters), each trimmed to a sane length."""
    lines = [l.strip()[:MAX_LINE_CHARS] for l in text.splitlines()]
    return [l for l in lines if len(l) >= 3][:max_lines]


_UNITS = {
    "minute": (timedelta(minutes=1), lambda t: t.replace(second=0, microsecond=0), "%H:%M"),
    "hour": (timedelta(hours=1), lambda t: t.replace(minute=0, second=0, microsecond=0), "%m-%d %H:00"),
    "day": (timedelta(days=1), lambda t: t.replace(hour=0, minute=0, second=0, microsecond=0), "%Y-%m-%d"),
}


def _line_order_timeline(items):
    total = len(items)
    n_buckets = min(10, total)
    size = math.ceil(total / n_buckets)
    out = []
    for b in range(n_buckets):
        chunk = items[b * size:(b + 1) * size]
        if not chunk:
            continue
        counts = Counter(i["label"] for i in chunk)
        out.append({"bucket": f"lines {b * size + 1}-{b * size + len(chunk)}", "total": len(chunk), "counts": dict(counts)})
    return out, "line"


def build_timeline(items):
    """Return (buckets, unit). Uses timestamps when at least half the lines have one."""
    if not items:
        return [], "line"
    stamped = [(datetime.fromisoformat(i["ts"]), i) for i in items if i.get("ts")]
    if len(stamped) * 2 < len(items):
        return _line_order_timeline(items)

    t0 = min(t for t, _ in stamped)
    t1 = max(t for t, _ in stamped)
    span = t1 - t0
    unit = "minute" if span <= timedelta(hours=2) else "hour" if span <= timedelta(days=3) else "day"
    step, trunc, fmt = _UNITS[unit]
    start = trunc(t0)
    n = int((trunc(t1) - start) / step) + 1
    if n > MAX_BUCKETS:
        return _line_order_timeline(items)

    buckets = [Counter() for _ in range(n)]
    for t, item in stamped:
        buckets[int((trunc(t) - start) / step)][item["label"]] += 1
    out = [
        {"bucket": (start + step * k).strftime(fmt), "total": sum(c.values()), "counts": dict(c)}
        for k, c in enumerate(buckets)
    ]
    return out, unit


def trend_alerts(items):
    """Flag categories that are clearly rising: compare the first half of the log with the second half."""
    n = len(items)
    if n < 6:
        return []
    ordered = sorted(items, key=lambda i: i["ts"]) if all(i.get("ts") for i in items) else items
    first, second = ordered[: n // 2], ordered[n // 2:]
    c1 = Counter(i["label"] for i in first if not i["unknown"])
    c2 = Counter(i["label"] for i in second if not i["unknown"])
    meta = {i["label"]: (i["title"], i["severity"]) for i in items}
    alerts = []
    for label, after in c2.items():
        before = c1.get(label, 0)
        if after >= 3 and after >= 3 * max(before, 1):
            title, severity = meta[label]
            alerts.append({
                "label": label, "title": title, "severity": severity, "before": before, "after": after,
                "message": f"{title} is rising: {before} in the first half of the log vs {after} in the second half.",
            })
    alerts.sort(key=lambda a: a["after"] - a["before"], reverse=True)
    return alerts


def summarize(items: list[dict]) -> dict:
    """items: [{line_no, text, label, title, severity, confidence, unknown, ts}]"""
    counts = Counter(i["label"] for i in items if not i["unknown"])
    titles = {i["label"]: i["title"] for i in items}
    timeline, unit = build_timeline(items)
    unknown = sum(1 for i in items if i["unknown"])
    return {
        "total": len(items),
        "failures": len(items) - unknown,
        "unknown": unknown,
        "by_label": [{"label": l, "title": titles[l], "count": n} for l, n in counts.most_common()],
        "timeline": timeline,
        "timeline_unit": unit,
        "alerts": trend_alerts(items),
    }
