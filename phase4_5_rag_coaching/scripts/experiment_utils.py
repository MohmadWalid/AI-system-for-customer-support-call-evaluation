"""
Shared utilities for experiment scripts.
"""
import json
from pathlib import Path

DIVIDER = "=" * 70


def print_summary(total: int, violations: int) -> None:
    """
    Prints a high-level experiment metrics summary.

    Args:
        total (int): Total number of transcripts evaluated.
        violations (int): Number of calls with violations.
    """
    print(DIVIDER)
    print(f"  Transcripts evaluated : {total}")
    print(f"  Calls with violations : {violations}")
    if total > 0:
        comp_rate = ((total - violations) / total) * 100
    else:
        comp_rate = 0.0
    print(f"  Compliance rate       : {comp_rate:.1f}%")
    print(DIVIDER)


def save_results(all_results: list, results_path) -> None:
    """
    Saves experiment results as JSON to the given path.

    Args:
        all_results (list): List of result dicts to save.
        results_path: Path object or string for the output file.
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {results_path.resolve()}")


def build_result(call_id, predicted_label, result) -> dict:
    """
    Builds a standard result dict from a call evaluation.

    Args:
        call_id: The call identifier string.
        predicted_label: The predicted fine intent label.
        result: The evaluation result dict from an evaluator.

    Returns:
        dict: Standardised result dictionary.
    """
    return {
        "call_id":              call_id,
        "fine_label_predicted": predicted_label,
        "verdict":              result.get("verdict"),
        "recovered":            result.get("recovered"),
        "recovery_note":        result.get("recovery_note"),
        "violations":           result.get("violations"),
        "overall_summary":      result.get("overall_summary"),
    }
