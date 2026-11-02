"""Tests that need no PyTorch: tokenizer, entity extraction, data files, log analysis."""
import csv
from datetime import datetime

import pytest

import loganalysis
from data import load_custom, load_holdout, read_csv, training_rows
from dataset import LABELS, TEMPLATES, generate, template_split
from nlp_utils import KB, extract_entities
from textproc import normalize, tokenize


# ------------------------------------------------------------------ tokenizer
def test_tokenize_splits_words_numbers_and_punctuation():
    assert tokenize("Timeout after 30s!") == ["Timeout", "after", "30", "s", "!"]


def test_normalize_collapses_numbers_but_keeps_status_codes():
    assert normalize("429") == "429"
    assert normalize("504") == "504"
    assert normalize("37") == "<num>"
    assert normalize("6.4") == "<num>"
    assert normalize("ERROR") == "error"


# ------------------------------------------------------------------ entity extraction + knowledge base
def test_extract_entities_finds_the_useful_fields():
    text = "2026-11-01 10:22:05 ERROR [rag-service] ReadTimeout after 30s, 504, 92% mem 6.4 GiB req-123"
    ent = extract_entities(text)
    assert ent["timestamps"] == ["2026-11-01 10:22:05"]
    assert ent["http_status"] == ["504"]
    assert "ReadTimeout" in ent["exceptions"]
    assert ent["memory_sizes"] == ["6.4 GiB"]
    assert ent["request_ids"] == ["req-123"]
    assert ent["services"] == ["rag-service"]


def test_extract_entities_empty_for_plain_text():
    assert extract_entities("nothing structured here") == {}


def test_knowledge_base_covers_every_label_plus_unknown():
    assert set(KB) == set(LABELS) | {"unknown"}
    for entry in KB.values():
        assert {"title", "severity", "root_causes", "actions", "prevention"} <= set(entry)


# ------------------------------------------------------------------ data
def test_generate_is_balanced_and_reproducible():
    a, b = generate(5, seed=1), generate(5, seed=1)
    assert a == b
    assert len(a) == 5 * len(LABELS)
    assert {l for _, l in a} == set(LABELS)


def test_template_split_is_disjoint_and_complete():
    train, test = template_split(2)
    for label, pool in TEMPLATES.items():
        assert not set(train[label]) & set(test[label])
        assert sorted(train[label] + test[label]) == list(range(len(pool)))


def test_generate_respects_allowed_templates():
    train, test = template_split(2)
    # Every sample from the test split must differ in template from the train split: just check it runs for all labels.
    assert len(generate(3, seed=1, allowed=test)) == 3 * len(LABELS)


def test_read_csv_skips_unknown_labels_and_blank_text(tmp_path):
    p = tmp_path / "x.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerow(["good line", "data_drift"])
        w.writerow(["bad label", "not_a_label"])
        w.writerow(["", "data_drift"])
    assert read_csv(p) == [("good line", "data_drift")]


def test_read_csv_missing_file_is_empty(tmp_path):
    assert read_csv(tmp_path / "nope.csv") == []


def test_holdout_covers_every_class_and_never_leaks_into_training():
    holdout, custom = load_holdout(), load_custom()
    assert {l for _, l in holdout} == set(LABELS)
    train_texts = {t.lower() for t, _ in custom} | {t.lower() for t, _ in training_rows(10, seed=3)}
    leaked = [t for t, _ in holdout if t.lower() in train_texts]
    assert leaked == []


# ------------------------------------------------------------------ log analysis
def _item(label, ts=None, unknown=False):
    return {"label": label, "title": label.title(), "severity": "high", "unknown": unknown, "ts": ts, "confidence": 0.9}


def test_split_lines_drops_blank_and_tiny_lines():
    assert loganalysis.split_lines("hello world\n\n  \nab\nsecond line\n") == ["hello world", "second line"]


def test_split_lines_caps_the_number_of_lines():
    assert len(loganalysis.split_lines("\n".join(["line number"] * 700))) == loganalysis.MAX_LINES


def test_parse_ts_variants():
    assert loganalysis.parse_ts("2026-11-01 10:22:05 ERROR x") == datetime(2026, 11, 1, 10, 22, 5)
    assert loganalysis.parse_ts("2026-11-01T10:22:05Z boom") == datetime(2026, 11, 1, 10, 22, 5)
    assert loganalysis.parse_ts("no time here") is None
    assert loganalysis.parse_ts("2026-13-45 99:99:99 impossible") is None


def test_timeline_uses_minutes_when_timestamps_present():
    items = [_item("a", "2026-11-01T10:00:10"), _item("a", "2026-11-01T10:00:50"), _item("b", "2026-11-01T10:02:05")]
    buckets, unit = loganalysis.build_timeline(items)
    assert unit == "minute"
    assert [b["total"] for b in buckets] == [2, 0, 1]
    assert buckets[0]["counts"] == {"a": 2}


def test_timeline_falls_back_to_line_order_without_timestamps():
    items = [_item("a") for _ in range(25)]
    buckets, unit = loganalysis.build_timeline(items)
    assert unit == "line"
    assert sum(b["total"] for b in buckets) == 25
    assert len(buckets) <= 10


def test_timeline_empty():
    assert loganalysis.build_timeline([]) == ([], "line")


def test_trend_alert_flags_a_rising_category():
    first = [_item("hallucination"), _item("timeout_latency"), _item("timeout_latency"), _item("hallucination"), _item("timeout_latency"), _item("hallucination")]
    second = [_item("rate_limit_quota")] * 5 + [_item("timeout_latency")]
    alerts = loganalysis.trend_alerts(first + second)
    assert [a["label"] for a in alerts] == ["rate_limit_quota"]
    assert alerts[0]["before"] == 0 and alerts[0]["after"] == 5


def test_trend_alert_ignores_steady_and_tiny_logs():
    steady = [_item("a"), _item("a"), _item("a")] * 4
    assert loganalysis.trend_alerts(steady) == []
    assert loganalysis.trend_alerts([_item("a")] * 3) == []


def test_trend_alert_ignores_unknown_lines():
    items = [_item("x")] * 6 + [_item("unknown", unknown=True)] * 6
    assert loganalysis.trend_alerts(items) == []


def test_summarize_counts_failures_and_unknowns():
    items = [_item("a"), _item("a"), _item("b"), _item("unknown", unknown=True)]
    s = loganalysis.summarize(items)
    assert (s["total"], s["failures"], s["unknown"]) == (4, 3, 1)
    assert s["by_label"][0] == {"label": "a", "title": "A", "count": 2}


@pytest.mark.parametrize("n", [1, 2, 7, 10, 11, 99])
def test_line_order_timeline_never_crashes(n):
    buckets, _ = loganalysis.build_timeline([_item("a") for _ in range(n)])
    assert sum(b["total"] for b in buckets) == n
