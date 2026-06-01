"""
Retriever that loads the full policy manual for a given intent,
prepended with baseline cross-cutting policies that apply to every call.
"""
from pathlib import Path
from config import MANUALS_DIR

BASELINE_DIR = Path(MANUALS_DIR) / "baseline"

def load(fine_label: str) -> str:
    """
    Reads the full text of the policy manual corresponding to a fine label,
    prepended with baseline policies that apply to every call regardless of intent.

    Args:
        fine_label (str): The predicted intent class of the call.

    Returns:
        str: Full text content of baseline + intent-specific policies.
    """
    manual_path = Path(MANUALS_DIR) / f"{fine_label}.txt"
    if not manual_path.exists():
        raise FileNotFoundError(f"No manual for class: {fine_label}")

    intent_text = manual_path.read_text(encoding="utf-8")

    # Prepend baseline policies if directory exists
    baseline_text = ""
    if BASELINE_DIR.exists():
        for baseline_file in sorted(BASELINE_DIR.glob("*.txt")):
            baseline_text += baseline_file.read_text(encoding="utf-8") + "\n"

    if baseline_text:
        return (
            f"[BASELINE POLICIES - Apply to every call]\n"
            f"{baseline_text}\n"
            f"[INTENT-SPECIFIC POLICIES: {fine_label}]\n"
            f"{intent_text}"
        )

    return intent_text
