"""
main.py — Wires classifier + RAG evaluator over test transcripts.
"""
import json
from pathlib import Path

from groq import Groq

from config import GROQ_API_KEY
from src.classifier import ClassifierPipeline
from scripts.transcripts import TRANSCRIPTS
from scripts.classify_transcript import classify
from scripts.retrievers import class_scoped
from scripts.evaluators import call_level
from scripts.experiment_utils import build_result
from src.scoring import score_all_calls

DIVIDER      = "=" * 70
RESULTS_PATH = Path("data/results.json")


def main():
    print(DIVIDER)
    print("  Phase 4 RAG — Compliance Evaluator")
    print(DIVIDER)

    print("\n[1/2] Loading classifier...")
    classifier = ClassifierPipeline()

    print(f"\n[2/2] Running {len(TRANSCRIPTS)} transcripts.\n")

    client = Groq(api_key=GROQ_API_KEY)

    total_violations = 0
    all_results      = []

    for transcript in TRANSCRIPTS:
        call_id        = transcript["call_id"]
        expected_label = transcript["fine_label"]
        utterances     = transcript["utterances"]

        # ── Classify: majority vote ───────────────────────────────────────
        predicted_label = classify(utterances, classifier)
        classifier_match = predicted_label == expected_label
        match_icon       = "OK" if classifier_match else "MISMATCH"

        print(DIVIDER)
        print(f"  Call ID  : {call_id}")
        print(f"  Expected : {expected_label}")
        print(f"  Predicted: {predicted_label}  [{match_icon}]")
        print(DIVIDER)

        try:
            policies = class_scoped.load(predicted_label)
        except FileNotFoundError:
            policies = "(no relevant policies retrieved above threshold)"
        result = call_level.evaluate(utterances, predicted_label, policies, client)

        print(f"  Verdict  : {result['verdict'].upper()}")
        print(f"  Summary  : {result['overall_summary']}")
        if result["violations"]:
            for v in result["violations"]:
                print(f"\n  !! VIOLATION (turn {v.get('turn')})")
                print(f"     Policy  : {v.get('violated_policy')}")
                print(f"     Evidence: {v.get('evidence')}")
                print(f"     Reason  : {v.get('reason')}")

        if result["verdict"] == "violation":
            total_violations += 1

        entry = build_result(call_id, predicted_label, result)
        entry["fine_label_expected"] = expected_label
        entry["classifier_match"]    = classifier_match
        all_results.append(entry)

        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print(DIVIDER)
    print("  SUMMARY")
    print(DIVIDER)
    print(f"  Transcripts evaluated : {len(TRANSCRIPTS)}")
    print(f"  Calls with violations : {total_violations}")
    print(f"  Compliance rate       : "
          f"{((len(TRANSCRIPTS) - total_violations) / len(TRANSCRIPTS) * 100):.1f}%")
    print(DIVIDER)

    # ── Save results ──────────────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Results saved to: {RESULTS_PATH.resolve()}")

    print("\n[Scoring] Computing call quality scores...")
    score_all_calls("data/results.json", TRANSCRIPTS)


if __name__ == "__main__":
    main()
