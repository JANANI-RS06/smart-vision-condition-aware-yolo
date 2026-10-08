"""Condition-specific preprocessing + ESRGAN super-resolution.

All enhancers take and return RGB uint8 numpy arrays.
Each one is a plain function, so you can swap in your own notebook implementation.
"""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- helpers

def _clahe_lab(rgb, clip=2.0, grid=8):
    l, a, b = cv2.split(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    )

    l = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=(grid, grid)
    ).apply(l)

    return cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2RGB
    )


def _guided_filter(guide, src, r, eps):
    """Edge-preserving smoothing using a box-filter guided filter."""

    box = lambda x: cv2.boxFilter(
        x,
        -1,
        (2 * r + 1, 2 * r + 1)
    )

    mg = box(guide)
    ms = box(src)

    var = box(guide * guide) - mg * mg
    cov = box(guide * src) - mg * ms

    a = cov / (var + eps)
    b = ms - a * mg

    return box(a) * guide + box(b)


def _dark_channel(img, size):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (size, size)
    )

    return cv2.erode(
        np.min(img, axis=2),
        kernel
    )


# ---------------------------------------------------------------- enhancers

def enhance_normal(rgb):
    return rgb


def enhance_low_light(rgb, gamma=1.8, clip=2.0):
    """Gamma correction + CLAHE."""

    table = (
        np.linspace(0, 1, 256)
        ** (1.0 / gamma)
        * 255
    ).astype(np.uint8)

    return _clahe_lab(
        cv2.LUT(rgb, table),
        clip
    )


def enhance_fog(
    rgb,
    omega=0.95,
    t0=0.1,
    patch=15,
    clip=2.0
):
    """Dark-channel-prior dehazing + CLAHE."""

    img = rgb.astype(np.float32) / 255.0

    dark = _dark_channel(img, patch)

    n = max(
        1,
        int(dark.size * 0.001)
    )

    idx = np.argpartition(
        dark.ravel(),
        -n
    )[-n:]

    atmo = np.maximum(
        img.reshape(-1, 3)[idx].mean(axis=0),
        1e-3
    )

    t = (
        1.0
        - omega
        * _dark_channel(
            img / atmo,
            patch
        )
    )

    gray = (
        cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        ).astype(np.float32)
        / 255.0
    )

    t = _guided_filter(
        gray,
        t.astype(np.float32),
        r=40,
        eps=1e-3
    )

    t = np.clip(
        t,
        t0,
        1.0
    )[..., None]

    out = np.clip(
        np.nan_to_num(
            ((img - atmo) / t + atmo) * 255
        ),
        0,
        255
    ).astype(np.uint8)

    return _clahe_lab(
        out,
        clip
    )


def enhance_rain(
    rgb,
    streak_width=5,
    thresh=18
):
    """Lightweight deraining using top-hat detection + inpainting."""

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (streak_width, 1)
    )

    tophat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        kernel
    )

    mask = (
        (tophat > thresh)
        .astype(np.uint8)
        * 255
    )

    mask = cv2.dilate(
        mask,
        np.ones((3, 3), np.uint8)
    )

    return cv2.inpaint(
        rgb,
        mask,
        3,
        cv2.INPAINT_TELEA
    )


def enhance_snow(
    rgb,
    clip=2.0
):
    """Snow/noise restoration: median + bilateral + CLAHE."""

    out = cv2.medianBlur(
        rgb,
        3
    )

    out = cv2.bilateralFilter(
        out,
        7,
        50,
        50
    )

    return _clahe_lab(
        out,
        clip
    )


def enhance_sand_dust(
    rgb,
    clip=2.0
):
    """Visibility enhancement + gray-world correction + CLAHE."""

    img = rgb.astype(np.float32)

    means = img.reshape(
        -1,
        3
    ).mean(axis=0)

    out = np.clip(
        img
        * (
            means.mean()
            / np.maximum(means, 1e-3)
        ),
        0,
        255
    ).astype(np.uint8)

    out = _clahe_lab(
        out,
        clip
    )

    return cv2.bilateralFilter(
        out,
        9,
        75,
        75
    )


# ---------------------------------------------------------------- enhancer registry

ENHANCERS = {
    "Normal": enhance_normal,
    "Low-Light": enhance_low_light,
    "Fog": enhance_fog,
    "Rain": enhance_rain,
    "Snow": enhance_snow,
    "Sand-Dust": enhance_sand_dust,
}


PROCESSING_LABELS = {
    "Normal": "None (original image)",
    "Low-Light": "Gamma correction + CLAHE",
    "Fog": "Fog dehazing + CLAHE",
    "Rain": "Deraining",
    "Snow": "Snow/noise restoration",
    "Sand-Dust": "Visibility enhancement + bilateral filtering",
}


def apply_condition(rgb, condition):
    """Run the enhancer for the selected condition.

    Returns:
        image, processing_label
    """

    fn = ENHANCERS.get(
        condition,
        enhance_normal
    )

    return (
        fn(rgb),
        PROCESSING_LABELS.get(
            condition,
            "None"
        )
    )


# ---------------------------------------------------------------- ESRGAN

def load_esrgan(cfg: dict):
    """Load Real-ESRGAN.

    Returns:
        (upsampler, error_message)

    Requires:
        basicsr
        realesrgan
        RealESRGAN_x4plus.pth
    """

    ecfg = cfg["esrgan"]

    if not ecfg.get("enabled", True):
        return (
            None,
            "ESRGAN is disabled in config.json."
        )

    path = ROOT / ecfg["weights"]

    if not path.exists():
        return (
            None,
            f"ESRGAN weights not found at {ecfg['weights']}."
        )

    try:
        import sys
        import torch
        import torchvision.transforms.functional as F

        # ---------------------------------------------------------
        # Compatibility fix for newer torchvision versions.
        # BasicSR expects this older module name.
        # ---------------------------------------------------------
        sys.modules[
            "torchvision.transforms.functional_tensor"
        ] = F

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        # ---------------------------------------------------------
        # Real-ESRGAN RRDB network
        # ---------------------------------------------------------

        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )

        # ---------------------------------------------------------
        # RealESRGAN upsampler
        # ---------------------------------------------------------

        up = RealESRGANer(
            scale=4,
            model_path=str(path),
            model=net,
            tile=ecfg.get("tile", 256),
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available()
        )

        return up, None

    except Exception as exc:
        return (
            None,
            f"ESRGAN could not be loaded: {exc}"
        )


def super_resolve(
    upsampler,
    rgb,
    ecfg: dict
):
    """Run Real-ESRGAN super-resolution.

    Large images are resized to max_input_side first
    to reduce memory usage.
    """

    h, w = rgb.shape[:2]

    limit = ecfg.get(
        "max_input_side",
        512
    )

    if max(h, w) > limit:

        s = limit / max(h, w)

        rgb = cv2.resize(
            rgb,
            (
                int(w * s),
                int(h * s)
            ),
            interpolation=cv2.INTER_AREA
        )

    # RGB → BGR for OpenCV / RealESRGAN
    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    out, _ = upsampler.enhance(
        bgr,
        outscale=ecfg.get(
            "scale",
            4
        )
    )

    # BGR → RGB
    return cv2.cvtColor(
        out,
        cv2.COLOR_BGR2RGB
    )