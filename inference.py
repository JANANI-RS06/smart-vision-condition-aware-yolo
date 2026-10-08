import sys
import time
import json
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from pipeline.classifier import load_classifier, predict_condition
from pipeline.gate import decide
from pipeline.enhancers import (
    apply_condition,
    load_esrgan,
    super_resolve
)
from pipeline.vlm import generate_explanation


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
CONFIG_PATH = BASE_DIR / "config.json"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD CONFIG
# ============================================================

if not CONFIG_PATH.exists():

    print("ERROR: config.json not found.")

    print(
        f"Expected location:\n{CONFIG_PATH}"
    )

    sys.exit(1)


with open(CONFIG_PATH, "r", encoding="utf-8") as f:

    CONFIG = json.load(f)


# ============================================================
# YOLO MODELS
# ============================================================

YOLO_MODELS = {
    "yolov8s": MODELS_DIR / "yolov8s.pt",
    "yolo11s": MODELS_DIR / "yolo11s.pt",
    "yolo26s": MODELS_DIR / "yolo26s.pt",
}


# ============================================================
# CONDITION CLASSIFIER
# ============================================================

def load_condition_classifier():

    print("\nLoading condition classifier...")

    bundle, error = load_classifier(CONFIG)

    if bundle is None:

        print("\nClassifier Error:")
        print(error)

        return None

    print(
        "Condition classifier loaded successfully."
    )

    print(
        f"Device           : {bundle['device']}"
    )

    return bundle


# ============================================================
# CONDITION CLASSIFICATION
# ============================================================

def classify_image(
    classifier_bundle,
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    condition, confidence, probabilities = (
        predict_condition(
            classifier_bundle,
            image
        )
    )

    return (
        condition,
        confidence,
        probabilities
    )


# ============================================================
# YOLO DETECTION
# ============================================================

def run_yolo(
    model_name,
    image_path
):

    model_path = YOLO_MODELS[
        model_name
    ]

    if not model_path.exists():

        raise FileNotFoundError(
            f"YOLO weights not found: "
            f"{model_path}"
        )

    print(
        f"\nLoading YOLO model: "
        f"{model_name}"
    )

    model = YOLO(
        str(model_path)
    )

    # --------------------------------------------------------
    # YOLO inference
    # --------------------------------------------------------

    results = model.predict(

        source=str(image_path),

        imgsz=640,

        conf=CONFIG.get(
            "detector",
            {}
        ).get(
            "conf",
            0.25
        ),

        iou=CONFIG.get(
            "detector",
            {}
        ).get(
            "iou",
            0.45
        ),

        verbose=False
    )

    result = results[0]

    detections = []

    # --------------------------------------------------------
    # Extract detected class names
    # --------------------------------------------------------

    if result.boxes is not None:

        for cls_id in result.boxes.cls:

            class_id = int(
                cls_id
            )

            class_name = model.names[
                class_id
            ]

            detections.append(
                class_name
            )

    return (
        result,
        detections
    )


# ============================================================
# SAVE YOLO RESULT
# ============================================================

def save_detection_result(
    result,
    model_name
):

    output_path = (
        RESULTS_DIR
        / f"{model_name}_result.jpg"
    )

    result.save(
        filename=str(
            output_path
        )
    )

    return output_path


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(

    model_name,

    image_path,

    condition,

    confidence,

    detections,

    inference_time,

    output_path

):

    csv_path = (
        RESULTS_DIR
        / f"{model_name}_results.csv"
    )

    new_file = not csv_path.exists()

    with open(
        csv_path,
        "a",
        encoding="utf-8"
    ) as f:

        if new_file:

            f.write(
                "image,model,condition,"
                "confidence,objects,"
                "inference_time_ms,"
                "output\n"
            )

        objects = "|".join(
            detections
        )

        f.write(

            f'"{image_path}",'

            f'"{model_name}",'

            f'"{condition}",'

            f'"{confidence:.4f}",'

            f'"{objects}",'

            f'"{inference_time:.2f}",'

            f'"{output_path}"\n'
        )

    return csv_path


# ============================================================
# CONDITION PREPROCESSING
# ============================================================

def run_condition_preprocessing(
    image_path,
    condition
):

    image = Image.open(
        image_path
    ).convert("RGB")

    rgb = np.array(
        image
    )

    processed_rgb, processing_label = (
        apply_condition(
            rgb,
            condition
        )
    )

    processed_path = (
        RESULTS_DIR
        / "processed_input.jpg"
    )

    Image.fromarray(
        processed_rgb
    ).save(
        processed_path
    )

    return (
        str(processed_path),
        processing_label
    )


# ============================================================
# ESRGAN
# ============================================================

def run_esrgan(
    image_path
):

    ecfg = CONFIG.get(
        "esrgan",
        {}
    )

    # --------------------------------------------------------
    # Check whether ESRGAN is enabled
    # --------------------------------------------------------

    if not ecfg.get(
        "enabled",
        True
    ):

        print(
            "ESRGAN           : Disabled"
        )

        return str(
            image_path
        )

    print(
        "\nLoading ESRGAN..."
    )

    # --------------------------------------------------------
    # Load ESRGAN
    # --------------------------------------------------------

    upsampler, error = (
        load_esrgan(
            CONFIG
        )
    )

    if upsampler is None:

        print(
            "ESRGAN           : Not available"
        )

        print(
            f"Reason           : {error}"
        )

        print(
            "Using processed image."
        )

        return str(
            image_path
        )

    # --------------------------------------------------------
    # Run ESRGAN
    # --------------------------------------------------------

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        rgb = np.array(
            image
        )

        sr_rgb = super_resolve(

            upsampler,

            rgb,

            ecfg
        )

        esrgan_path = (
            RESULTS_DIR
            / "esrgan_output.jpg"
        )

        Image.fromarray(
            sr_rgb
        ).save(
            esrgan_path,
            quality=95
        )

        print(
            "ESRGAN           : Applied"
        )

        print(
            f"ESRGAN Output    : "
            f"{esrgan_path}"
        )

        return str(
            esrgan_path
        )

    except Exception as e:

        print(
            f"ESRGAN           : Failed"
        )

        print(
            f"Reason           : {e}"
        )

        print(
            "Using processed image."
        )

        return str(
            image_path
        )


# ============================================================
# MAIN INFERENCE
# ============================================================

def inference(
    image_path,
    model_name
):

    start_time = (
        time.perf_counter()
    )

    image_path = Path(
        image_path
    )

    # ========================================================
    # CHECK INPUT IMAGE
    # ========================================================

    if not image_path.exists():

        print(
            "\nERROR: Image not found:"
        )

        print(
            image_path
        )

        return

    # ========================================================
    # CHECK YOLO MODEL
    # ========================================================

    if model_name not in YOLO_MODELS:

        print(
            "\nERROR: Invalid YOLO model."
        )

        print(
            "\nAvailable models:"
        )

        for name in YOLO_MODELS:

            print(
                f"  - {name}"
            )

        return

    # ========================================================
    # HEADER
    # ========================================================

    print("\n")

    print(
        "=" * 55
    )

    print(
        "              SMART VISION"
    )

    print(
        "=" * 55
    )

    print(
        f"\nInput Image      : "
        f"{image_path}"
    )

    # ========================================================
    # STEP 1
    # CONDITION CLASSIFICATION
    # ========================================================

    print(
        "\n[1/5] Condition Classification"
    )

    classifier_bundle = (
        load_condition_classifier()
    )

    if classifier_bundle is None:

        return

    try:

        (
            condition,
            confidence,
            probabilities
        ) = classify_image(

            classifier_bundle,

            image_path
        )

    except Exception as e:

        print(
            "\nClassifier error:"
        )

        print(
            e
        )

        return

    print(
        f"Condition        : "
        f"{condition}"
    )

    print(
        f"Confidence       : "
        f"{confidence * 100:.1f}%"
    )

    # --------------------------------------------------------
    # Class probabilities
    # --------------------------------------------------------

    print(
        "\nCondition Probabilities:"
    )

    for (
        class_name,
        probability
    ) in probabilities.items():

        print(

            f"  {class_name:<12} : "
            f"{probability * 100:.2f}%"
        )

    # ========================================================
    # STEP 2
    # CONFIDENCE GATE
    # ========================================================

    print(
        "\n[2/5] Confidence Gate"
    )

    threshold = CONFIG.get(
        "confidence_threshold",
        0.80
    )

    gate_decision = decide(

        condition,

        confidence,

        threshold
    )

    processing_applied = (
        gate_decision.apply_processing
    )

    print(
        f"Threshold        : "
        f"{threshold * 100:.0f}%"
    )

    print(
        f"Gate Decision    : "
        f"{gate_decision.label}"
    )

    print(
        f"Reason           : "
        f"{gate_decision.reason}"
    )

    # ========================================================
    # STEP 3
    # CONDITION PREPROCESSING
    # ========================================================

    print(
        "\n[3/5] Condition-Specific Preprocessing"
    )

    processed_image = str(
        image_path
    )

    preprocessing_name = (
        "Original image"
    )

    if processing_applied:

        try:

            (
                processed_image,
                preprocessing_name
            ) = run_condition_preprocessing(

                image_path,

                condition
            )

            print(
                f"Preprocessing    : "
                f"{preprocessing_name}"
            )

        except Exception as e:

            print(
                f"Preprocessing    : "
                f"Failed ({e})"
            )

            print(
                "Using original image."
            )

            processed_image = str(
                image_path
            )

    else:

        print(
            "Preprocessing    : Skipped"
        )

    # ========================================================
    # ESRGAN
    # ========================================================

    if processing_applied:

        esrgan_image = run_esrgan(
            processed_image
        )

    else:

        print(
            "ESRGAN           : Skipped"
        )

        esrgan_image = str(
            image_path
        )

    # ========================================================
    # STEP 4
    # YOLO DETECTION
    # ========================================================

    print(
        "\n[4/5] Object Detection"
    )

    print(
        f"YOLO Detector    : "
        f"{model_name}"
    )

    try:

        (
            result,
            detections
        ) = run_yolo(

            model_name,

            esrgan_image
        )

    except Exception as e:

        print(
            "\nYOLO error:"
        )

        print(
            e
        )

        return

    print(
        f"Objects Detected : "
        f"{len(detections)}"
    )

    if detections:

        print(
            "Detected Objects : "
            + ", ".join(
                detections
            )
        )

    else:

        print(
            "Detected Objects : None"
        )

    # ========================================================
    # SAVE YOLO OUTPUT
    # ========================================================

    output_path = (
        save_detection_result(
            result,
            model_name
        )
    )

    # ========================================================
    # STEP 5
    # VLM EXPLANATION
    # ========================================================

    print(
        "\n[5/5] Vision-Language Model"
    )

    print(
        "VLM              : "
        "SmolVLM-500M-Instruct"
    )

    print(
        "\nGenerating explanation..."
    )

    try:

        vlm_explanation = (
            generate_explanation(

                esrgan_image,

                condition=condition,

                detections=detections
            )
        )

    except Exception as e:

        vlm_explanation = (
            "VLM explanation could not "
            "be generated."
        )

        print(
            "\nVLM Error:"
        )

        print(
            e
        )

    # ========================================================
    # TOTAL INFERENCE TIME
    # ========================================================

    end_time = (
        time.perf_counter()
    )

    inference_time = (

        end_time - start_time

    ) * 1000

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n")

    print(
        "=" * 55
    )

    print(
        "                 FINAL RESULT"
    )

    print(
        "=" * 55
    )

    print(
        f"\nInput Image      : "
        f"{image_path}"
    )

    print(
        f"\nCondition        : "
        f"{condition}"
    )

    print(
        f"Confidence       : "
        f"{confidence * 100:.1f}%"
    )

    print(
        f"\nConfidence Gate  : "
        f"{gate_decision.label}"
    )

    print(
        f"Gate Reason      : "
        f"{gate_decision.reason}"
    )

    print(
        f"Preprocessing    : "
        f"{preprocessing_name}"
    )

    if processing_applied:

        print(
            "ESRGAN           : "
            "Applied"
        )

    else:

        print(
            "ESRGAN           : "
            "Skipped"
        )

    print(
        f"\nYOLO Detector    : "
        f"{model_name}"
    )

    print(
        f"Objects Detected : "
        f"{len(detections)}"
    )

    if detections:

        print(
            "Detected Objects : "
            + ", ".join(
                detections
            )
        )

    print(
        "\nVLM Explanation:"
    )

    print(
        vlm_explanation
    )

    print(
        f"\nInference Time   : "
        f"{inference_time:.2f} ms"
    )

    print(
        f"\nOutput           : "
        f"{output_path}"
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = save_csv(

        model_name=model_name,

        image_path=image_path,

        condition=condition,

        confidence=confidence,

        detections=detections,

        inference_time=inference_time,

        output_path=output_path
    )

    print(
        f"CSV              : "
        f"{csv_path}"
    )

    print(
        "\n" + "=" * 55
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    if (
        len(sys.argv) == 2
        and sys.argv[1] == "--evaluate"
    ):

        print(
            "\nEvaluation mode is handled "
            "by the classifier evaluation "
            "pipeline."
        )

        return

    # --------------------------------------------------------
    # Check arguments
    # --------------------------------------------------------

    if len(sys.argv) < 3:

        print(
            "\nUsage:"
        )

        print(

            '.\\venv\\Scripts\\python.exe '
            'inference.py '
            '"samples\\image1.png" '
            'yolo11s'
        )

        print(
            "\nAvailable models:"
        )

        for name in YOLO_MODELS:

            print(
                f"  - {name}"
            )

        return

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    image_path = sys.argv[1]

    model_name = (
        sys.argv[2].lower()
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    inference(

        image_path,

        model_name
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()