"""Central place for folder locations. Environment variables let tests use temporary folders."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"


def artifacts_dir() -> Path:
    path = Path(os.environ.get("AIFSI_ARTIFACTS", ROOT / "artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path
