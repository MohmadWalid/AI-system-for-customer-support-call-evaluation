"""
src/runtime_rag.py — Phase 2: Runtime retrieval & compliance evaluation.

For every agent utterance in a transcript:
  1. Embed the utterance (same model used for indexing).
  2. Query FAISS for top-k most similar policy chunks.
  3. Compute confidence: High if best score >= threshold, else Low.
  4. Send utterance + retrieved policies to Groq LLM.
  5. Parse LLM response into a structured ComplianceResult.

Usage:
  python -m src.runtime_rag
"""

import json
import os
import re
import sys

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    GROQ_API_KEY,
    GROQ_MODEL,
    POLICY_MAP_PATH,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from data.transcripts import TRANSCRIPTS


# ── data classes (plain dicts for simplicity) ──────────────────────────────────

def make_result(
    call_id: str,
    utterance: str,
    violation: str,           # "Yes" | "No"
    violated_policy: str,     # policy name or "N/A"
    matched_policy_id: str,   # policy ID  or "N/A"
    reason: str,
    evidence: str,
    confidence: str,          # "High" | "Low"
) -> dict:
    return {
        "call_id": call_id,
        "utterance": utterance,
        "violation": violation,
        "violated_policy": violated_policy,
        "matched_policy_id": matched_policy_id,
        "reason": reason,
        "evidence": evidence,
        "confidence": confidence,
    }


# ── RAG core ──────────────────────────────────────────────────────────────────

class RAGEvaluator:
    """Wraps embedding model, FAISS index, policy map, and Groq client."""

    def __init__(self) -> None:
        print(f"[runtime] Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        print(f"[runtime] Loading FAISS index from: {FAISS_INDEX_PATH}")
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at '{FAISS_INDEX_PATH}'. "
                "Run `python -m src.setup_index` first."
            )
        self.index: faiss.IndexFlatIP = faiss.read_index(FAISS_INDEX_PATH)

        print(f"[runtime] Loading policy map from: {POLICY_MAP_PATH}")
        with open(POLICY_MAP_PATH, "r", encoding="utf-8") as f:
            self.policy_map: list[dict] = json.load(f)

        print("[runtime] Initialising Groq client …")
        self.groq_client = Groq(api_key=GROQ_API_KEY)

    # ── retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, utterance: str) -> tuple[list[dict], float, str]:
        """
        Embed utterance, query FAISS, return (top_policies, best_score, confidence).

        Returns:
          top_policies  – list of policy metadata dicts (up to TOP_K, deduplicated)
          best_score    – highest cosine similarity found
          confidence    – "High" | "Low"
        """
        embedding: np.ndarray = self.embedder.encode(
            [utterance],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(embedding, TOP_K)
        scores = scores[0]      # shape (TOP_K,)
        indices = indices[0]    # shape (TOP_K,)

        best_score: float = float(scores[0])
        confidence: str = "High" if best_score >= SIMILARITY_THRESHOLD else "Low"


        # Deduplicate by policy_id (multiple chunks from same policy)
        seen_ids: set[str] = set()
        top_policies: list[dict] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.policy_map):
                continue
            meta = self.policy_map[idx].copy()
            meta["retrieval_score"] = float(score)
            pid = meta["policy_id"]
            if pid not in seen_ids:
                seen_ids.add(pid)
                top_policies.append(meta)

        return top_policies, best_score, confidence

    # ── LLM evaluation ────────────────────────────────────────────────────────

    def _build_prompt(self, utterance: str, policies: list[dict]) -> str:
        policy_block = "\n\n".join(
            f"[{p['policy_id']}] {p['policy_name']} ({p['policy_tag']})\n{p['policy_text']}"
            for p in policies
        )
        return f"""You are a financial compliance auditor. Your job is to decide whether an agent utterance violates any of the retrieved compliance policies.

## Agent Utterance
\"\"\"{utterance}\"\"\"

## Retrieved Policies
{policy_block}

## Task
Analyse the utterance strictly against the policies above.
Respond ONLY with a valid JSON object in this exact schema — no markdown fences, no extra text:

{{
  "violation": "Yes" or "No",
  "violated_policy": "<Policy Name or N/A>",
  "matched_policy_id": "<Policy ID or N/A>",
  "reason": "<one-sentence explanation>",
  "evidence": "<exact quote from the utterance that constitutes the violation, or N/A>"
}}"""

    def _call_llm(self, prompt: str) -> dict:
        """Send prompt to Groq and parse JSON response."""
        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw: str = response.choices[0].message.content.strip()

        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE).strip()
        raw = re.sub(r"```$", "", raw, flags=re.MULTILINE).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Graceful fallback
            return {
                "violation": "No",
                "violated_policy": "N/A",
                "matched_policy_id": "N/A",
                "reason": "LLM returned unparseable response.",
                "evidence": "N/A",
            }

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate_utterance(self, call_id: str, utterance: str) -> dict:
        """Full RAG pipeline for a single utterance."""
        top_policies, best_score, confidence = self.retrieve(utterance)

        if not top_policies:
            return make_result(
                call_id=call_id,
                utterance=utterance,
                violation="No",
                violated_policy="N/A",
                matched_policy_id="N/A",
                reason="No relevant policies retrieved.",
                evidence="N/A",
                confidence="Low",
            )

        prompt = self._build_prompt(utterance, top_policies)
        llm_output = self._call_llm(prompt)

        return make_result(
            call_id=call_id,
            utterance=utterance,
            violation=llm_output.get("violation", "No"),
            violated_policy=llm_output.get("violated_policy", "N/A"),
            matched_policy_id=llm_output.get("matched_policy_id", "N/A"),
            reason=llm_output.get("reason", "N/A"),
            evidence=llm_output.get("evidence", "N/A"),
            confidence=confidence,
        )

    def evaluate_transcript(self, transcript: dict) -> list[dict]:
        """Evaluate all agent utterances in a transcript."""
        call_id: str = transcript["call_id"]
        results: list[dict] = []
        for utt in transcript["utterances"]:
            if utt["speaker"] != "agent":
                continue   # only evaluate agent speech
            result = self.evaluate_utterance(call_id, utt["text"])
            results.append(result)
        return results


# ── entry point ───────────────────────────────────────────────────────────────

def run_all_transcripts() -> list[dict]:
    evaluator = RAGEvaluator()
    all_results: list[dict] = []

    for transcript in TRANSCRIPTS:
        print(f"\n[runtime] ── {transcript['call_id']} ({transcript['agent']}) ──")
        results = evaluator.evaluate_transcript(transcript)
        for r in results:
            all_results.append(r)
            _print_result(r)

    return all_results


def _print_result(r: dict) -> None:
    print(f"  Utterance : {r['utterance'][:80]}{'…' if len(r['utterance']) > 80 else ''}")
    print(f"  Violation : {r['violation']}")
    if r["violation"] == "Yes":
        print(f"  Policy    : [{r['matched_policy_id']}] {r['violated_policy']}")
        print(f"  Evidence  : {r['evidence']}")
        print(f"  Reason    : {r['reason']}")
    print(f"  Confidence: {r['confidence']}")
    print()


if __name__ == "__main__":
    results = run_all_transcripts()

    # Optionally save results to JSON
    output_path = "data/results.json"
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[runtime] Results saved → {output_path}")
