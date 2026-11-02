"""Fair evaluation and model comparison.   python ml/evaluate.py   [--bert] [--epochs 8] [--n 400]

Three experiments, from easiest (optimistic) to hardest (honest):
  random_split       random 80/20 split of synthetic logs  -> usually ~100%, misleading
  heldout_templates  train on some templates, test on templates never seen
  handwritten        train on all synthetic + custom_logs.csv, test on holdout_logs.csv

Writes artifacts/report.json and artifacts/report.md (the app's 'Overview' tab reads report.json).
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from common import predict_proba  # noqa: E402
from data import load_holdout, training_rows  # noqa: E402
from dataset import LABELS, generate, template_split  # noqa: E402
from paths import artifacts_dir  # noqa: E402
from textproc import normalize, tokenize  # noqa: E402
from train import train_gru  # noqa: E402

MODEL_NAMES = {
    "tfidf_logreg": "Simple baseline (TF-IDF + Logistic Regression)",
    "bigru": "The app's model",
    "distilbert": "DistilBERT (fine-tuned)",
}


def _tok(text):
    return [normalize(t) for t in tokenize(text)]


def tfidf_predict(train_rows, test_rows):
    pipe = make_pipeline(
        TfidfVectorizer(tokenizer=_tok, token_pattern=None, lowercase=False, ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=5.0),
    )
    pipe.fit([t for t, _ in train_rows], [l for _, l in train_rows])
    return list(pipe.predict([t for t, _ in test_rows]))


def make_gru_predict(epochs):
    def run(train_rows, test_rows):
        model, vocab = train_gru(train_rows, epochs=epochs, verbose=False)
        probs = predict_proba(model, vocab, [t for t, _ in test_rows])
        return [LABELS[i] for i in probs.argmax(1).tolist()]
    return run


def bert_predict(train_rows, test_rows, epochs=2, max_train=1500, max_len=64, bs=16):
    """Optional and slow on CPU. Needs: pip install transformers"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(LABELS))
    l2i = {l: i for i, l in enumerate(LABELS)}
    rows = list(train_rows)
    random.Random(0).shuffle(rows)
    rows = rows[:max_train]
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for ep in range(epochs):
        random.Random(ep).shuffle(rows)
        for i in range(0, len(rows), bs):
            b = rows[i:i + bs]
            enc = tok([t for t, _ in b], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            out = model(**enc, labels=torch.tensor([l2i[l] for _, l in b]))
            out.loss.backward()
            opt.step()
            opt.zero_grad()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_rows), 64):
            b = test_rows[i:i + 64]
            enc = tok([t for t, _ in b], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            preds += model(**enc).logits.argmax(1).tolist()
    return [LABELS[p] for p in preds]


def score(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
        "per_class": {
            l: {
                "precision": round(rep[l]["precision"], 4),
                "recall": round(rep[l]["recall"], 4),
                "f1": round(rep[l]["f1-score"], 4),
                "support": int(rep[l]["support"]),
            }
            for l in LABELS
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def write_markdown(report, path):
    models = list(next(iter(report["experiments"].values()))["models"].keys())
    lines = ["# Model evaluation report", "", f"Generated: {report['generated_at']}", ""]
    lines += ["| Experiment | Train | Test | " + " | ".join(f"{MODEL_NAMES[m]} (acc / macro-F1)" for m in models) + " |"]
    lines += ["|---|---|---|" + "---|" * len(models)]
    for exp in report["experiments"].values():
        cells = [f"{exp['models'][m]['accuracy']:.1%} / {exp['models'][m]['macro_f1']:.3f}" for m in models]
        lines.append(f"| {exp['title']} | {exp['n_train']} | {exp['n_test']} | " + " | ".join(cells) + " |")
    if "handwritten" in report["experiments"]:
        exp = report["experiments"]["handwritten"]
        lines += ["", "## Per-class results on the handwritten holdout (the app's model)", "",
                  "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for l, s in exp["models"]["bigru"]["per_class"].items():
            lines.append(f"| {l} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | {s['support']} |")
    lines += ["", "Note: the 'random split' row is optimistic because train and test logs share the same templates."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", action="store_true", help="also fine-tune DistilBERT (slow, needs transformers)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n", type=int, default=400, help="synthetic samples per class")
    a = ap.parse_args()

    experiments = {}
    synth = generate(a.n, seed=42)
    cut = int(len(synth) * 0.8)
    experiments["random_split"] = ("Random split of synthetic logs (optimistic)", synth[:cut], synth[cut:])

    train_allowed, test_allowed = template_split(2)
    experiments["heldout_templates"] = (
        "Held-out templates (unseen wording patterns)",
        training_rows(a.n, seed=1, allowed=train_allowed, include_custom=False),
        generate(max(60, a.n // 4), seed=2, allowed=test_allowed),
    )

    holdout = load_holdout()
    if holdout:
        experiments["handwritten"] = (
            "Handwritten holdout (real-style wording)",
            training_rows(a.n, seed=42),
            holdout,
        )
    else:
        print("datasets/holdout_logs.csv not found or empty - skipping the handwritten experiment")

    runners = {"tfidf_logreg": tfidf_predict, "bigru": make_gru_predict(a.epochs)}
    if a.bert:
        runners["distilbert"] = bert_predict

    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "labels": LABELS, "experiments": {}}
    for key, (title, train_rows, test_rows) in experiments.items():
        print(f"\n== {title}: {len(train_rows)} train / {len(test_rows)} test")
        y_true = [l for _, l in test_rows]
        results = {}
        for mkey, run in runners.items():
            t0 = time.time()
            res = score(y_true, run(train_rows, test_rows))
            results[mkey] = res
            print(f"   {MODEL_NAMES[mkey]:<32} accuracy={res['accuracy']:.1%}  macro-F1={res['macro_f1']:.3f}  ({time.time() - t0:.0f}s)")
        report["experiments"][key] = {"title": title, "n_train": len(train_rows), "n_test": len(test_rows), "models": results}

    art = artifacts_dir()
    (art / "report.json").write_text(json.dumps(report), encoding="utf-8")
    write_markdown(report, art / "report.md")
    print(f"\nSaved {art / 'report.json'} and {art / 'report.md'}")
    print("Open the app's 'Overview' tab to see the table and confusion matrix.")


if __name__ == "__main__":
    main()
