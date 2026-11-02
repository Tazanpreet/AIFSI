"""Test setup: make ml/ and backend/ importable and give every test a throw-away database."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "ml", ROOT / "backend"):
    sys.path.insert(0, str(p))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point storage at an empty temporary SQLite file."""
    monkeypatch.setenv("AIFSI_DB", str(tmp_path / "test.db"))
    import storage

    storage.init()
    return storage
