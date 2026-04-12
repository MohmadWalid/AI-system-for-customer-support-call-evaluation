"""
main.py — Top-level entry point for the Phase 4 RAG Compliance System.

Steps:
  1. Build the FAISS index from the policy dataset  (setup_index)
  2. Run the RAG evaluator over all mock transcripts (runtime_rag)

Usage:
  python main.py
"""

from src.setup_index import build_index
from src.runtime_rag import run_all_transcripts


def main() -> None:
    print("=" * 60)
    print("  PHASE 4 — RAG Compliance System")
    print("=" * 60)

    print("\n[Step 1/2] Building FAISS policy index …\n")
    build_index()

    print("\n[Step 2/2] Running RAG evaluation on transcripts …\n")
    run_all_transcripts()

    print("\n" + "=" * 60)
    print("  Done. Results saved to data/results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
