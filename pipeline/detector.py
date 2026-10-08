"""Object detection with YOLOv8s, YOLO11s and YOLO26s."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# Detection result
# ============================================================

@dataclass
class DetectionResult:
    annotated: object
    table: pd.DataFrame
    count: int
    inference_ms: float


# ============================================================
# Single YOLO detector
# ============================================================

def load_detector(cfg: dict):
    """Load the main detector specified in config.json."""

    path = ROOT / cfg["detector"]["weights"]

    if not path.exists():
        return None, (
            "Detector weights are not available.\n"
            f"Missing file: {path}"
        )

    try:
        from ultralytics import YOLO

        model = YOLO(str(path))
        return model, None

    except Exception as exc:
        return None, f"Detector could not be loaded: {exc}"


def run_detection(
    model,
    rgb,
    conf=0.25,
    iou=0.45
) -> DetectionResult:
    """Run one YOLO model on an RGB image."""

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    t0 = time.perf_counter()

    results = model.predict(
        bgr,
        conf=conf,
        iou=iou,
        verbose=False
    )

    ms = (time.perf_counter() - t0) * 1000

    res = results[0]

    rows = []

    for box in res.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)

        rows.append(
            {
                "Object": res.names[class_id],
                "Confidence": confidence
            }
        )

    table = pd.DataFrame(
        rows,
        columns=["Object", "Confidence"]
    )

    if not table.empty:
        table = (
            table
            .sort_values("Confidence", ascending=False)
            .reset_index(drop=True)
        )

    annotated = cv2.cvtColor(
        res.plot(),
        cv2.COLOR_BGR2RGB
    )

    return DetectionResult(
        annotated=annotated,
        table=table,
        count=len(table),
        inference_ms=ms
    )


# ============================================================
# THREE YOLO MODELS
# ============================================================

YOLO_MODELS = {
    "YOLOv8s": "models/yolov8s.pt",
    "YOLO11s": "models/yolo11s.pt",
    "YOLO26s": "models/yolo26s.pt",
}


def load_all_detectors():
    """
    Load YOLOv8s, YOLO11s and YOLO26s.

    Returns:
        models: dictionary containing successfully loaded models
        errors: dictionary containing loading errors
    """

    from ultralytics import YOLO

    models = {}
    errors = {}

    for name, relative_path in YOLO_MODELS.items():

        path = ROOT / relative_path

        if not path.exists():
            errors[name] = f"Missing: {path}"
            continue

        try:
            models[name] = YOLO(str(path))
            print(f"✅ {name} loaded")

        except Exception as exc:
            errors[name] = str(exc)
            print(f"❌ {name}: {exc}")

    return models, errors


def compare_detectors(
    rgb,
    models,
    conf=0.25,
    iou=0.45
):
    """
    Run all loaded YOLO models on the SAME image.

    Returns:
        comparison_results:
            {
                "YOLOv8s": DetectionResult,
                "YOLO11s": DetectionResult,
                "YOLO26s": DetectionResult
            }

        summary:
            pandas DataFrame containing:
            Model
            Detections
            Best Confidence
            Average Confidence
            Inference (ms)
    """

    comparison_results = {}
    summary_rows = []

    for name, model in models.items():

        result = run_detection(
            model=model,
            rgb=rgb,
            conf=conf,
            iou=iou
        )

        comparison_results[name] = result

        if result.table.empty:
            best_confidence = 0.0
            average_confidence = 0.0
        else:
            best_confidence = float(
                result.table["Confidence"].max()
            )

            average_confidence = float(
                result.table["Confidence"].mean()
            )

        summary_rows.append(
            {
                "Model": name,
                "Detections": result.count,
                "Best Confidence": best_confidence,
                "Average Confidence": average_confidence,
                "Inference (ms)": result.inference_ms
            }
        )

    summary = pd.DataFrame(summary_rows)

    return comparison_results, summary


# ============================================================
# Detection reliability
# ============================================================

GOOD = "GOOD"
RECOVERY = "RECOVERY APPLIED"
NOT_CONNECTED = "NOT CONNECTED"


@dataclass
class Reliability:
    status: str
    message: str
    result: Optional[DetectionResult] = None


_checker: Optional[Callable] = None


def register_reliability_checker(fn: Callable):
    """
    Register the detection reliability checker.

    fn(image_rgb, detection_result, context)
    must return a dictionary.
    """

    global _checker

    _checker = fn


def check_reliability(
    image_rgb,
    result: DetectionResult,
    context: dict
) -> Reliability:

    if _checker is None:

        return Reliability(
            NOT_CONNECTED,
            (
                "Reliability logic is not connected yet. "
                "Register a checker with "
                "register_reliability_checker() "
                "in pipeline/detector.py."
            )
        )

    try:

        out = _checker(
            image_rgb,
            result,
            context
        )

        status = (
            RECOVERY
            if out.get("status") == "recovery"
            else GOOD
        )

        return Reliability(
            status,
            out.get("message", ""),
            out.get("result")
        )

    except Exception as exc:

        return Reliability(
            NOT_CONNECTED,
            f"Reliability check failed: {exc}"
        )