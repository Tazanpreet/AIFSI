"""Pure-Python text processing (no PyTorch needed): tokenizer and normalizer."""
import re

PAD, UNK = "<pad>", "<unk>"
MAX_LEN = 96

TOKEN_RE = re.compile(r"[A-Za-z_]+|\d+(?:\.\d+)?|[^\sA-Za-z\d]")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
STATUS_RE = re.compile(r"[1-5]\d\d")


def tokenize(text: str) -> list[str]:
    """Split raw log text into word / number / punctuation tokens."""
    return TOKEN_RE.findall(text)


def normalize(tok: str) -> str:
    """Lowercase; collapse numbers to <num> but keep HTTP-like status codes (429, 504...)."""
    t = tok.lower()
    if STATUS_RE.fullmatch(t):
        return t
    if NUM_RE.fullmatch(t):
        return "<num>"
    return t
