"""
Experiment: Call-level evaluation + Class-scoped index.
Condition 1 of the 2x2 ablation matrix.
Saves results to data/experiments/call_class_results.json
"""
import sys
from pathlib import Path

# Adjust system path to import modules from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import math
from groq import Groq

from config import GROQ_API_KEY
from scripts.transcripts import TRANSCRIPTS
from src.classifier import ClassifierPipeline
from scripts.retrievers import class_scoped
from scripts.evaluators import call_level
from scripts.classify_transcript import classify
from scripts.experiment_utils import print_summary, save_results, build_result, DIVIDER

# Define experiment directory and output file paths
EXPERIMENTS_DIR  = Path("data/experiments")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-gt', action='store_true', help='Use ground truth intent labels instead of classifier')
    parser.add_argument('--sample', type=int, default=None, help='Sample N transcripts proportionally across quality levels')
    args = parser.parse_args()

    if args.sample is not None:
        RESULTS_PATH = EXPERIMENTS_DIR / "call_class_gt_sample_results.json"
    elif args.use_gt:
        RESULTS_PATH = EXPERIMENTS_DIR / "call_class_gt_v2_results.json"
    else:
        RESULTS_PATH = EXPERIMENTS_DIR / "call_class_results.json"

    print(DIVIDER)
    print("  Experiment 1: Call-Level + Class-Scoped Index")
    print(DIVIDER)

    transcripts = list(TRANSCRIPTS)  # make a mutable copy

    if args.sample is not None:
        from collections import defaultdict
        # Group by quality level
        by_quality = defaultdict(list)
        for t in transcripts:
            by_quality[t["quality_level"]].append(t)
        
        total = len(transcripts)
        sampled = []
        for quality, group in sorted(by_quality.items()):
            # Proportional allocation, at least 1 per quality level
            k = max(1, round(args.sample * len(group) / total))
            sampled.extend(sorted(group, key=lambda x: x["call_id"])[:k])
        
        # Trim to exact sample size if rounding pushed over
        transcripts = sorted(sampled, key=lambda x: x["call_id"])[:args.sample]
        print(f"[sample] Using {len(transcripts)} transcripts (sampled from {total})")

    # Load components
    client     = Groq(api_key=GROQ_API_KEY)
    classifier = ClassifierPipeline() if not args.use_gt else None

    total_violations = 0
    all_results      = []

    # Run evaluations over transcripts
    for transcript in transcripts:
        call_id    = transcript["call_id"]
        utterances = transcript["utterances"]

        # Majority vote classification per transcript
        predicted_label = transcript["intent"] if args.use_gt else classify(utterances, classifier)

        # 1) Retrieve policies using class-scoped
        try:
            policies = class_scoped.load(predicted_label)
        except FileNotFoundError:
            policies = "(no relevant policies retrieved above threshold)"

        # 2) Evaluate using call-level
        result = call_level.evaluate(
            utterances, predicted_label, policies, client
        )

        verdict_display = result.get("verdict", "ERROR").upper()
        print(f"{call_id[:50]:<50} {verdict_display:>10}")

        if result.get("verdict") == "violation":
            total_violations += 1

        all_results.append(build_result(call_id, predicted_label, result))

    print_summary(len(transcripts), total_violations)
    save_results(all_results, RESULTS_PATH)


if __name__ == "__main__":
    main()
