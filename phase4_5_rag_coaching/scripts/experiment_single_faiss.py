"""
scripts/experiment_single_faiss.py — Experiment 1: Single global FAISS index.
Runs all 64 transcripts through one global index instead of class-scoped indexes.
Query = fine_label + customer turns joined → top 5 chunks above threshold 0.45
Saves results to data/experiments/single_faiss_results.json
"""
import json
import sys
import time
from pathlib import Path

# Adjust system path to import modules from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

from config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL, SIMILARITY_THRESHOLD, TOP_K
from scripts.transcripts import TRANSCRIPTS
from src.classifier import ClassifierPipeline

# Define experiment directory and output file paths
EXPERIMENTS_DIR  = Path("data/experiments")
INDEX_PATH       = EXPERIMENTS_DIR / "single_index.faiss"
MAP_PATH         = EXPERIMENTS_DIR / "single_map.json"
RESULTS_PATH     = EXPERIMENTS_DIR / "single_faiss_results.json"

DIVIDER          = "=" * 70


def retrieve(query: str, embedder, index, policy_map: list) -> list[dict]:
    """
    Retrieves top K semantically similar policy rules from the single global index based on a unified query.
    
    Args:
        query (str): The search query composed of predicted label and customer utterances.
        embedder: Loaded SentenceTransformer model.
        index: Loaded FAISS index from disk.
        policy_map (list): Source rules mapped to their index positions.

    Returns:
        list[dict]: List of matching policy rules with similarity scores.
    """
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
            "rule":   entry["rule"],
            "source": entry["source"],
            "score":  round(float(score), 4),
        })
    return results


def evaluate_call(utterances: list[dict], fine_label: str,
                  embedder, index, policy_map: list,
                  client: Groq) -> dict:
    """
    Evaluates agent compliance across a single call conversation using global retrieval.

    Args:
        utterances (list[dict]): A list of all dialogue turns (speaker, text).
        fine_label (str): The predicted fine intent category.
        embedder: Loaded SentenceTransformer model.
        index: Loaded FAISS index.
        policy_map (list): Loaded rules map.
        client (Groq): Groq API client.

    Returns:
        dict: Evaluation results containing verdict, recovered status, recovery note, violations, and summary.
    """
    # Step 1: build query from fine_label + customer turns
    customer_text = " ".join(
        u["text"] for u in utterances if u["speaker"] == "customer"
    )
    query = f"{fine_label} {customer_text}"

    # Step 2: retrieve top 5 from single global index
    retrieved = retrieve(query, embedder, index, policy_map)

    if retrieved:
        policies_text = "\n".join(f"- {r['source']}" for r in retrieved)
    else:
        policies_text = "(no relevant policies retrieved above threshold)"

    # Step 3: build interleaved conversation
    conversation = "\n".join(
        f"[{i+1}] {u['speaker'].capitalize()}: \"{u['text']}\""
        for i, u in enumerate(utterances)
    )

    # Construct the evaluation prompt for final call outcomes
    prompt = (
        f"You are a banking call center QA evaluator.\n"
        f"Issue class: {fine_label}\n\n"
        f"CONVERSATION:\n{conversation}\n\n"
        f"POLICIES:\n{policies_text}\n\n"
        f"Evaluate the agent's FINAL performance in this conversation.\n"
        f"Focus on the outcome of the call, not individual turns.\n"
        f"If the agent made a mistake early but corrected it before the call ended, "
        f"do NOT flag it as a violation.\n"
        f"Only flag violations that were UNRESOLVED at the end of the call.\n\n"
        f"Reply ONLY in this JSON format:\n"
        f'{{"verdict": "violation" or "ok", '
        f'"recovered": true or false, '
        f'"recovery_note": "one sentence or empty string", '
        f'"violations": [{{"turn": 1, "violated_policy": "...", '
        f'"evidence": "...", "reason": "..."}}], '
        f'"overall_summary": "one sentence about the agent\'s overall performance"}}'
    )

    # Groq API rate-limit resilient retry loop (max 9 attempts)
    for attempt in range(1, 10):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            break
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e) or (
                hasattr(e, "status_code") and e.status_code == 429
            ):
                print(f"  [Groq] Rate limit hit. Sleeping 5s (attempt {attempt}/9)...")
                time.sleep(5)
            else:
                raise e

    # Robust JSON parsing from LLM completion response
    raw = response.choices[0].message.content.strip()
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        parsed = {"verdict": "error", "violations": [], "overall_summary": raw}

    return {
        "verdict":         parsed.get("verdict"),
        "recovered":       parsed.get("recovered", False),
        "recovery_note":   parsed.get("recovery_note", ""),
        "violations":      parsed.get("violations", []),
        "overall_summary": parsed.get("overall_summary", ""),
        "rules_retrieved": len(retrieved),
    }


def main():
    if not INDEX_PATH.exists():
        print("[ERROR] Single index not found. Run scripts/build_single_index.py first.")
        sys.exit(1)

    print(DIVIDER)
    print("  Experiment 1: Single FAISS Index")
    print(DIVIDER)

    print("\nLoading single index...")
    index      = faiss.read_index(str(INDEX_PATH))
    policy_map = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    print(f"Index loaded: {index.ntotal} vectors")

    # Load components
    embedder   = SentenceTransformer(EMBEDDING_MODEL)
    client     = Groq(api_key=GROQ_API_KEY)
    classifier = ClassifierPipeline()

    total_violations = 0
    all_results      = []

    # Run evaluations over transcripts
    for transcript in TRANSCRIPTS:
        call_id    = transcript["call_id"]
        utterances = transcript["utterances"]

        # Majority vote classification per transcript
        label_counts = {}
        label_scores = {}
        for u in utterances:
            if u["speaker"] == "customer":
                prediction = classifier(u["text"])
                lbl   = prediction["fine_label"]
                score = prediction.get("confidence", 0.0)
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
                label_scores[lbl] = label_scores.get(lbl, 0.0) + score

        predicted_label = max(
            label_counts.keys(),
            key=lambda k: (label_counts[k], label_scores[k])
        ) if label_counts else "unknown"

        result = evaluate_call(
            utterances, predicted_label,
            embedder, index, policy_map, client
        )

        verdict_display = result["verdict"].upper() if result["verdict"] else "ERROR"
        print(f"{call_id[:50]:<50} {verdict_display:>10}  rules={result['rules_retrieved']}")

        if result["verdict"] == "violation":
            total_violations += 1

        all_results.append({
            "call_id":              call_id,
            "fine_label_predicted": predicted_label,
            "verdict":              result["verdict"],
            "recovered":            result["recovered"],
            "recovery_note":        result["recovery_note"],
            "violations":           result["violations"],
            "overall_summary":      result["overall_summary"],
            "rules_retrieved":      result["rules_retrieved"],
        })

    # Display high-level experiment metrics summary
    print(DIVIDER)
    print(f"  Transcripts evaluated : {len(TRANSCRIPTS)}")
    print(f"  Calls with violations : {total_violations}")
    print(f"  Compliance rate       : "
          f"{((len(TRANSCRIPTS) - total_violations) / len(TRANSCRIPTS) * 100):.1f}%")
    print(DIVIDER)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
