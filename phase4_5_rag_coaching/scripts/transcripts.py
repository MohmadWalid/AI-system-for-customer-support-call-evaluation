"""
src/transcripts.py — loads pre-generated transcripts from data/transcripts/*.json.

Each dict is augmented with backwards-compatible aliases so the rest of the
pipeline (main.py, scoring.py) can keep using 'fine_label' and 'utterances'
without modification.
"""
import json
from collections import Counter
from pathlib import Path

_TRANSCRIPT_DIR = Path(__file__).parent.parent / "data" / "transcripts"


def _load() -> list[dict]:
    if not _TRANSCRIPT_DIR.exists() or not _TRANSCRIPT_DIR.is_dir():
        print(f"[transcripts] ERROR: directory not found: {_TRANSCRIPT_DIR}")
        return []

    files = sorted(_TRANSCRIPT_DIR.glob("*.json"))
    if not files:
        print(f"[transcripts] ERROR: no .json files found in {_TRANSCRIPT_DIR}")
        return []

    loaded = []
    for f in files:
        try:
            t = json.loads(f.read_text(encoding="utf-8"))
            # backwards-compatible aliases for existing pipeline code
            t["fine_label"] = t["intent"]
            t["utterances"] = t["transcript"]
            loaded.append(t)
        except Exception as exc:
            print(f"[transcripts] WARNING: could not load {f.name}: {exc}")

    counts = Counter(t["quality_level"] for t in loaded)
    noise  = sum(1 for t in loaded if t.get("noise_injected"))
    print(
        f"[transcripts] Loaded {len(loaded)} transcripts | "
        + " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        + f" | noise: {noise}"
    )
    return loaded


TRANSCRIPTS: list[dict] = _load()
