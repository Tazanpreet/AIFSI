"""Data loading: synthetic logs + your own CSV files.

datasets/custom_logs.csv   -> extra TRAINING examples (add your own here)
datasets/holdout_logs.csv  -> handwritten TEST examples (never used for training)
CSV format: two columns named  text,label
"""
import csv
import random
import sys
from pathlib import Path

from dataset import LABELS, generate
from paths import DATASETS

CUSTOM_REPEAT = 8  # how many times custom rows are repeated so they matter next to 3600 synthetic rows


def read_csv(path) -> list[tuple[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for line_no, r in enumerate(csv.DictReader(f), start=2):
            text = (r.get("text") or "").strip()
            label = (r.get("label") or "").strip()
            if not text:
                continue
            if label not in LABELS:
                print(f"[data] {path.name} line {line_no}: unknown label {label!r}, skipped", file=sys.stderr)
                continue
            rows.append((text, label))
    return rows


def load_custom():
    return read_csv(DATASETS / "custom_logs.csv")


def load_holdout():
    return read_csv(DATASETS / "holdout_logs.csv")


def training_rows(n_per_class: int = 400, seed: int = 42, allowed=None, include_custom: bool = True):
    rows = generate(n_per_class, seed, allowed)
    if include_custom:
        rows += load_custom() * CUSTOM_REPEAT
    random.Random(seed).shuffle(rows)
    return rows
