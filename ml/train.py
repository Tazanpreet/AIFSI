"""Train the BiGRU failure classifier.  Run from the project root:  python ml/train.py"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from common import FailureClassifier, encode, predict_proba  # noqa: E402
from data import load_custom, load_holdout, training_rows  # noqa: E402
from dataset import LABELS  # noqa: E402
from paths import artifacts_dir  # noqa: E402
from textproc import PAD, UNK, normalize, tokenize  # noqa: E402

WORD_DROPOUT = 0.1  # randomly hide 10% of words while training -> teaches the model what <unk> means


def build_vocab(rows) -> dict[str, int]:
    counts = Counter(normalize(t) for text, _ in rows for t in tokenize(text))
    vocab = {PAD: 0, UNK: 1}
    for tok in counts:
        vocab.setdefault(tok, len(vocab))
    return vocab


def train_gru(rows, epochs: int = 8, seed: int = 7, batch: int = 64, lr: float = 3e-3, verbose: bool = True):
    """Train on (text, label) rows. Returns (model, vocab)."""
    torch.manual_seed(seed)
    random.seed(seed)
    vocab = build_vocab(rows)
    label2id = {l: i for i, l in enumerate(LABELS)}
    x = torch.tensor([encode(tokenize(t), vocab) for t, _ in rows])
    y = torch.tensor([label2id[l] for _, l in rows])

    model = FailureClassifier(len(vocab), len(LABELS))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), batch):
            idx = perm[i:i + batch]
            xb = x[idx]
            hide = (torch.rand(xb.shape) < WORD_DROPOUT) & (xb != 0)
            xb = xb.masked_fill(hide, vocab[UNK])
            opt.zero_grad()
            loss = loss_fn(model(xb), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose:
            print(f"epoch {epoch}/{epochs}  loss={total / len(x):.4f}")
    model.eval()
    return model, vocab


def accuracy(model, vocab, rows):
    if not rows:
        return None
    pred = predict_proba(model, vocab, [t for t, _ in rows]).argmax(1).tolist()
    return sum(LABELS[p] == l for p, (_, l) in zip(pred, rows)) / len(rows)


def save_artifacts(model, vocab) -> Path:
    art = artifacts_dir()
    torch.save({"state_dict": model.state_dict(), "vocab_size": len(vocab), "labels": LABELS}, art / "model.pt")
    (art / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    return art


def main():
    rows = training_rows(n_per_class=400)
    n_custom = len(load_custom())
    print(f"Training on {len(rows)} rows ({n_custom} custom examples from datasets/custom_logs.csv, repeated)")
    model, vocab = train_gru(rows)
    art = save_artifacts(model, vocab)
    print(f"Saved model and vocab to {art}")

    holdout = load_holdout()
    acc = accuracy(model, vocab, holdout)
    if acc is not None:
        print(f"Accuracy on the {len(holdout)} handwritten holdout logs (never trained on): {acc:.1%}")
    print("Run 'python ml/evaluate.py' for a full fair comparison with a confusion matrix.")


if __name__ == "__main__":
    main()
