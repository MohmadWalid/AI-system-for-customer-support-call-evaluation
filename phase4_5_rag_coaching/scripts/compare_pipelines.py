"""
scripts/compare_pipelines.py — Qualitative comparison between
class-scoped (full manual) vs single FAISS (top 5 chunks) pipelines.
Shows exactly which policies each pipeline feeds to the LLM for 5 selected calls.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, MANUALS_DIR, SIMILARITY_THRESHOLD, TOP_K
from scripts.transcripts import TRANSCRIPTS

EXPERIMENTS_DIR = Path("data/experiments")
INDEX_PATH      = EXPERIMENTS_DIR / "single_index.faiss"
MAP_PATH        = EXPERIMENTS_DIR / "single_map.json"
DIVIDER         = "=" * 70
DIVIDER_SMALL   = "-" * 70

TARGET_CALLS = [
    "cancel_transfer_bad_1",
    "card_arrival_recovery_1",
    "balance_not_updated_good_1",
    "transfer_not_received_ambiguous_1",
    "disputed_charge_bad_2",
]


def load_manual(fine_label: str) -> str:
    manual_path = Path(MANUALS_DIR) / f"{fine_label}.txt"
    if not manual_path.exists():
        return f"(manual not found for {fine_label})"
    return manual_path.read_text(encoding="utf-8")


def retrieve_single(query: str, embedder, index, policy_map: list) -> list[dict]:
    vec = embedder.encode([query], normalize_embeddings=True)
    vec = np.array(vec, dtype=np.float32)

    k_actual = min(TOP_K, index.ntotal)
    scores, indices = index.search(vec, k_actual)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if float(score) < SIMILARITY_THRESHOLD:
            continue
        entry = policy_map[idx]
        results.append({
            "intent": entry["intent"],
            "source": entry["source"],
            "score":  round(float(score), 4),
        })
    return results


def main():
    if not INDEX_PATH.exists():
        print("[ERROR] Single index not found. Run scripts/build_single_index.py first.")
        sys.exit(1)

    index      = faiss.read_index(str(INDEX_PATH))
    policy_map = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    embedder   = SentenceTransformer(EMBEDDING_MODEL)

    transcript_map = {t["call_id"]: t for t in TRANSCRIPTS}

    output_lines = []

    for call_id in TARGET_CALLS:
        transcript = transcript_map.get(call_id)
        if transcript is None:
            output_lines.append(f"[WARNING] Call not found: {call_id}")
            continue

        utterances = transcript["utterances"]
        fine_label = transcript["fine_label"]

        customer_text = " ".join(
            u["text"] for u in utterances if u["speaker"] == "customer"
        )
        query = f"{fine_label} {customer_text}"

        retrieved = retrieve_single(query, embedder, index, policy_map)

        correct = sum(1 for r in retrieved if r["intent"] == fine_label)
        wrong   = len(retrieved) - correct

        lines = []
        lines.append(DIVIDER)
        lines.append(f"  Call ID  : {call_id}")
        lines.append(f"  Intent   : {fine_label}")
        lines.append(f"  Quality  : {transcript.get('quality_level', 'unknown')}")
        lines.append(DIVIDER)
        lines.append(f"\n[ SINGLE FAISS ] Retrieved {len(retrieved)} chunks:")
        lines.append(DIVIDER_SMALL)

        if retrieved:
            for i, r in enumerate(retrieved, 1):
                match  = "✅" if r["intent"] == fine_label else "❌"
                intent = r["intent"][:35].ljust(35)
                rule   = r["source"][:60].split("**")
                # extract rule name — text between ** markers if present
                rule_name = rule[1] if len(rule) >= 3 else r["source"][:60]
                rule_name = rule_name.strip()[:55]
                lines.append(f"  [{i}] {match} {intent} ({r['score']}) {rule_name}")
        else:
            lines.append("  (no chunks retrieved above threshold)")

        lines.append("")
        lines.append(f"  Correct intent chunks : {correct}/{len(retrieved)}")
        lines.append(f"  Wrong intent chunks   : {wrong}/{len(retrieved)}")
        lines.append(DIVIDER_SMALL)

        output_lines.extend(lines)

    output_lines.append(f"\n{DIVIDER}")
    output_lines.append("  Comparison complete.")
    output_lines.append(DIVIDER)

    # Print to console
    output = "\n".join(output_lines)
    print(output)

    # Save to file
    out_path = Path("data/experiments/pipeline_comparison.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(f"\nSaved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
