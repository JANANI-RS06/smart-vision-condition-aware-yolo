"""Confidence gate: decide between condition-specific processing and the original image."""
from dataclasses import dataclass

PROCESSING_APPLIED = "PROCESSING APPLIED"
FALLBACK = "FALLBACK TO ORIGINAL"


@dataclass
class GateDecision:
    apply_processing: bool
    label: str
    reason: str
    threshold: float


def decide(condition, confidence, threshold: float) -> GateDecision:
    """If confidence >= threshold and condition != Normal -> apply preprocessing, else use original."""
    if condition is None or confidence is None:
        return GateDecision(False, FALLBACK, "Classifier output is not available.", threshold)
    if condition == "Normal":
        return GateDecision(False, FALLBACK, "Condition is Normal, so no processing is needed.", threshold)
    if confidence < threshold:
        return GateDecision(
            False, FALLBACK,
            f"Confidence {confidence:.1%} is below the {threshold:.0%} threshold.", threshold)
    return GateDecision(
        True, PROCESSING_APPLIED,
        f"Confidence {confidence:.1%} meets the {threshold:.0%} threshold.", threshold)
