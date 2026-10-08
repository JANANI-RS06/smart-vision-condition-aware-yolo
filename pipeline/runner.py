"""Runs the full Smart Vision pipeline. No Streamlit imports, so app.py, batch mode
and make_samples.py all share one code path.

    classify -> gate -> condition preprocessing -> ESRGAN -> YOLO -> reliability check
"""
import hashlib
import time

from . import classifier, detector, enhancers, gate

_CACHE = {}  # small memo so slider changes don't redo the slow stages


def _memo(key, fn):
    if key in _CACHE:
        return _CACHE[key]
    val = fn()
    if len(_CACHE) >= 12:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = val
    return val


def _timed(fn):
    t0 = time.perf_counter()
    val = fn()
    return val, (time.perf_counter() - t0) * 1000


def load_models(cfg: dict) -> dict:
    """Each entry is (model_or_None, error_message_or_None)."""
    return {
        "classifier": classifier.load_classifier(cfg),
        "esrgan": enhancers.load_esrgan(cfg),
        "detector": detector.load_detector(cfg),
    }


def run(rgb, pil, models: dict, cfg: dict, settings: dict = None) -> dict:
    s = {
        "threshold": cfg["confidence_threshold"],
        "det_conf": cfg["detector"]["conf"],
        "use_esrgan": cfg["esrgan"].get("enabled", True),
        "baseline": True,
    }
    s.update(settings or {})
    h = hashlib.md5(rgb.tobytes()).hexdigest()
    out = {"original": rgb, "timings": {}, "settings": s}

    # 1. condition classifier
    cls, cls_err = models["classifier"]
    if cls:
        (cond, conf, probs), ms = _memo(("cls", h), lambda: _timed(lambda: classifier.predict_condition(cls, pil)))
        out.update(condition=cond, confidence=conf, probs=probs)
        out["timings"]["classify"] = ms
    else:
        out.update(condition=None, confidence=None, probs=None, cls_error=cls_err)

    # 2. confidence gate
    g = gate.decide(out["condition"], out["confidence"], s["threshold"])
    out["gate"] = g

    # 3. condition-specific preprocessing
    if g.apply_processing:
        cond = out["condition"]
        (pre, label), ms = _memo(("pre", h, cond), lambda: _timed(lambda: enhancers.apply_condition(rgb, cond)))
        out["timings"]["preprocess"] = ms
    else:
        pre, label = rgb, "None"
    out["pre"], out["processing"] = pre, label

    # 4. ESRGAN (after preprocessing, before detection)
    esr, esr_err = models["esrgan"]
    wants = g.apply_processing or cfg["esrgan"].get("apply_to_original", False)
    processed = pre
    if not s["use_esrgan"]:
        out["esrgan"] = "Disabled (sidebar)"
    elif not wants:
        out["esrgan"] = "Skipped (original image path)"
    elif esr is None:
        out["esrgan"], out["esrgan_error"] = "Not available", esr_err
    else:
        key = ("sr", h, out["condition"] if g.apply_processing else "orig")
        processed, ms = _memo(key, lambda: _timed(lambda: enhancers.super_resolve(esr, pre, cfg["esrgan"])))
        out["esrgan"] = "Applied"
        out["timings"]["esrgan"] = ms
    out["processed"] = processed

    # 5. detection (+ optional baseline on the untouched original)
    det, det_err = models["detector"]
    out.update(detection=None, baseline=None, reliability=None)
    if det is None:
        out["det_error"] = det_err
        return out
    dcfg = cfg["detector"]
    out["detection"] = detector.run_detection(det, processed, s["det_conf"], dcfg["iou"])
    out["timings"]["detect"] = out["detection"].inference_ms
    if s["baseline"]:
        out["baseline"] = detector.run_detection(det, rgb, s["det_conf"], dcfg["iou"])
        out["timings"]["baseline"] = out["baseline"].inference_ms

    # 6. reliability / recovery
    ctx = {"condition": out["condition"], "confidence": out["confidence"], "gate": g,
           "detector": det, "config": cfg}
    rel = detector.check_reliability(processed, out["detection"], ctx)
    out["reliability"] = rel
    if rel.status == detector.RECOVERY and rel.result is not None:
        out["detection"] = rel.result
    return out
