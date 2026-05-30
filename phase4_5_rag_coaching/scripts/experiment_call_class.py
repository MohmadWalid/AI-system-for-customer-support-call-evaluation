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
    args = parser.parse_args()

    RESULTS_PATH = EXPERIMENTS_DIR / ("call_class_gt_results.json" if args.use_gt else "call_class_results.json")

    print(DIVIDER)
    print("  Experiment 1: Call-Level + Class-Scoped Index")
    print(DIVIDER)

    # Load components
    client     = Groq(api_key=GROQ_API_KEY)
    classifier = ClassifierPipeline() if not args.use_gt else None

    total_violations = 0
    all_results      = []

    # Run evaluations over transcripts
    for transcript in TRANSCRIPTS:
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

    print_summary(len(TRANSCRIPTS), total_violations)
    save_results(all_results, RESULTS_PATH)


if __name__ == "__main__":
    main()
