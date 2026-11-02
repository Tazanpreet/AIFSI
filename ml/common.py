"""Shared model code: encoding, the BiGRU classifier, and batch prediction."""
import torch
import torch.nn as nn

from textproc import MAX_LEN, PAD, UNK, normalize, tokenize  # noqa: F401  (re-exported)


def encode(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    ids = [vocab.get(normalize(t), vocab[UNK]) for t in tokens[:MAX_LEN]]
    return ids + [vocab[PAD]] * (MAX_LEN - len(ids))


class FailureClassifier(nn.Module):
    """Embedding -> BiGRU -> max-pool -> Linear. Small enough to train on a CPU in seconds."""

    def __init__(self, vocab_size: int, n_classes: int, emb: int = 64, hid: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hid * 2, n_classes)

    def embed(self, x):
        """Fixed-size vector for each text (used for 'similar past incidents')."""
        h, _ = self.gru(self.emb(x))
        mask = (x != 0).unsqueeze(-1)
        h = h.masked_fill(~mask, -1e9)
        return h.max(dim=1).values

    def forward(self, x):
        return self.fc(self.drop(self.embed(x)))


@torch.no_grad()
def predict_proba(model, vocab, texts: list[str], batch: int = 128) -> torch.Tensor:
    """Class probabilities, shape (len(texts), n_classes)."""
    model.eval()
    outs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        x = torch.tensor([encode(tokenize(t), vocab) for t in chunk])
        outs.append(torch.softmax(model(x), dim=1))
    return torch.cat(outs) if outs else torch.zeros((0, 0))
