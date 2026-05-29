"""
scripts/build_single_index.py — Builds a single global FAISS index
combining all 78 policy manuals into one index.
Saves to data/experiments/single_index.faiss and data/experiments/single_map.json
"""
import json
import sys
from pathlib import Path

# Adjust system path to import modules from the parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, MANUALS_DIR

# Define outputs paths for single index experiment
EXPERIMENTS_DIR = Path("data/experiments")
INDEX_PATH      = EXPERIMENTS_DIR / "single_index.faiss"
MAP_PATH        = EXPERIMENTS_DIR / "single_map.json"

# Text chunking hyperparameter: number of sentences per chunk
CHUNK_SIZE = 3  # sentences per chunk


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Splits manual text into small chunks of sentences to serve as vector database passages.
    
    Args:
        text (str): Full text of the policy manual.
        chunk_size (int): Number of sentences to group together in a single chunk.
        
    Returns:
        list[str]: List of text chunks with ending periods.
    """
    # Remove newlines and split by periods to extract individual sentences
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    chunks = []
    
    # Group sentences together in batches of size chunk_size
    for i in range(0, len(sentences), chunk_size):
        chunk = ". ".join(sentences[i:i + chunk_size]) + "."
        chunks.append(chunk)
    return chunks


def main():
    # Create the experiments directory if it doesn't already exist
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize SentenceTransformer embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    manuals_path = Path(MANUALS_DIR)

    all_chunks = []
    policy_map = []

    # Locate and sort all manual files (.txt) to ensure deterministic order
    manual_files = sorted(manuals_path.glob("*.txt"))
    print(f"Found {len(manual_files)} manuals. Building single index...")

    # Process each policy manual file and chunk its content
    for manual_file in manual_files:
        intent = manual_file.stem  # Intent is derived from the manual file name
        text   = manual_file.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        # Map each chunk to its corresponding intent, a short summary rule, and the raw chunk text
        for chunk in chunks:
            policy_map.append({
                "intent": intent,
                "rule":   chunk[:80],  # Short visual preview of the rule
                "source": chunk,       # Full sentence-level content
            })
            all_chunks.append(chunk)

    print(f"Total chunks generated: {len(all_chunks)}")

    # Encode all chunks using SentenceTransformer to get high-dimensional vector embeddings
    print("Encoding chunks...")
    embeddings = embedder.encode(all_chunks, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Build a flat FAISS Index with Inner Product (Cosine Similarity since embeddings are normalized)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Persist the FAISS index and the source chunk map to disk
    faiss.write_index(index, str(INDEX_PATH))
    MAP_PATH.write_text(json.dumps(policy_map, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Single index saved: {INDEX_PATH}  ({index.ntotal} vectors)")
    print(f"Map saved: {MAP_PATH}")


if __name__ == "__main__":
    main()
