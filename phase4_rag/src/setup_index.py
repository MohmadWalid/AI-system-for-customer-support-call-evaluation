"""
src/setup_index.py — Phase 1: Data ingestion and FAISS index building.

Steps:
  1. Load the mock policy dataset.
  2. Chunk policy text if it exceeds a length threshold.
  3. Embed each chunk with sentence-transformers (all-MiniLM-L6-v2).
  4. Build a FAISS flat-L2 index and add all embeddings.
  5. Persist the FAISS index and a JSON mapping (index → policy metadata).

Run once (or whenever policies change):
  python -m src.setup_index
"""

import json
import os
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── imports from this project ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    POLICY_MAP_PATH,
)
from data.policies import POLICIES

# ── constants ──────────────────────────────────────────────────────────────────
MAX_CHUNK_CHARS: int = 400      # characters; chunks longer than this get split
CHUNK_OVERLAP_CHARS: int = 50   # overlap between consecutive chunks


# ── helpers ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """
    Split *text* into overlapping chunks of at most *max_chars* characters.
    If the text is shorter than *max_chars* it is returned as-is.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap   # slide back by overlap

    return chunks


def build_policy_chunks(policies: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Expand the policy list into (chunks, metadata) pairs.

    Returns:
      texts    – flat list of chunk strings to embed
      metadata – parallel list of dicts containing the source policy info
    """
    texts: list[str] = []
    metadata: list[dict] = []

    for policy in policies:
        chunks = chunk_text(policy["text"])
        for chunk_idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadata.append({
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "policy_id": policy["id"],
                "policy_name": policy["name"],
                "policy_tag": policy["tag"],
                "policy_text": policy["text"],   # full text for LLM context
            })

    return texts, metadata


# ── main ──────────────────────────────────────────────────────────────────────

def build_index() -> None:
    print(f"[setup] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"[setup] Building policy chunks from {len(POLICIES)} policies …")
    texts, metadata = build_policy_chunks(POLICIES)
    print(f"[setup]   → {len(texts)} chunks total")

    print("[setup] Generating embeddings …")
    sys.stdout.flush()              # flush all prints before tqdm appears

    # sentence-transformers passes show_progress_bar straight to tqdm.
    # By temporarily swapping stdout ↔ stderr we ensure the progress bar
    # goes to stderr while our structured [setup] lines stay on stdout.
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr         # tqdm writes to sys.stdout by default
    try:
        embeddings: np.ndarray = model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise → cosine via inner product
        )
    finally:
        sys.stdout = _real_stdout   # always restore, even on error
    sys.stderr.flush()              # ensure bar line finishes before next print

    # Build FAISS index (Inner-Product on L2-normalised vectors ≡ cosine sim)
    dim: int = embeddings.shape[1]
    print(f"[setup] Building FAISS IndexFlatIP (dim={dim}) …")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"[setup]   → {index.ntotal} vectors indexed")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    # Persist index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"[setup] FAISS index saved → {FAISS_INDEX_PATH}")

    # Persist policy map
    with open(POLICY_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[setup] Policy map saved  → {POLICY_MAP_PATH}")


if __name__ == "__main__":
    build_index()
