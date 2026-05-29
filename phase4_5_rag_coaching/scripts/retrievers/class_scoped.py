"""
Retriever that loads the full policy manual for a given intent.
"""
from pathlib import Path
from config import MANUALS_DIR

def load(fine_label: str) -> str:
    """
    Reads the full text of the policy manual corresponding to a fine label.

    Args:
        fine_label (str): The predicted intent class of the call.

    Returns:
        str: Full text content of the manual.
    """
    manual_path = Path(MANUALS_DIR) / f"{fine_label}.txt"
    if not manual_path.exists():
        raise FileNotFoundError(f"No manual for class: {fine_label}")
    return manual_path.read_text(encoding="utf-8")
