"""
Grad-CAM colorization helpers for the production web viewer.

Turns the explainer agent's raw 2D activation map into clinician-friendly PNG
layers (base MRI, colorized heatmap, and a blended overlay) encoded as base64 so
the radiology-style frontend can render an adjustable heatmap over the scan.

Uses only numpy + Pillow (no matplotlib) to stay Windows-ARM64 friendly.
"""

from __future__ import annotations

import base64
import io
from typing import Dict

import numpy as np
from PIL import Image

# A compact "jet"-like colormap (blue -> cyan -> green -> yellow -> red) as RGB
# control points; clinically intuitive for activation intensity.
_JET = np.array(
    [
        [0, 0, 128],
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 128, 0],
        [255, 0, 0],
    ],
    dtype=np.float32,
)


def _apply_jet(norm: np.ndarray) -> np.ndarray:
    """Map a HxW array in [0,1] to an HxWx3 uint8 RGB image via the jet ramp."""
    x = np.clip(norm, 0.0, 1.0) * (len(_JET) - 1)
    lo = np.floor(x).astype(int)
    hi = np.clip(lo + 1, 0, len(_JET) - 1)
    frac = (x - lo)[..., None]
    rgb = _JET[lo] * (1 - frac) + _JET[hi] * frac
    return rgb.astype(np.uint8)


def _to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_gradcam(
    base_image: Image.Image,
    heatmap: np.ndarray,
    size: int = 384,
    alpha: float = 0.45,
) -> Dict[str, str]:
    """
    Build base / heatmap / overlay PNGs (base64) from a grayscale MRI and a
    Grad-CAM activation map.

    Returns dict with keys ``base``, ``heatmap``, ``overlay`` (data already
    base64-encoded, no data-URI prefix).
    """
    base = base_image.convert("L").resize((size, size), Image.BICUBIC)
    base_rgb = base.convert("RGB")

    # Normalize + upscale the activation map to the display resolution.
    hm = np.asarray(heatmap, dtype=np.float32)
    denom = hm.max() - hm.min()
    hm = (hm - hm.min()) / denom if denom > 1e-8 else np.zeros_like(hm)
    hm_img = Image.fromarray((hm * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    hm_up = np.asarray(hm_img, dtype=np.float32) / 255.0

    color = _apply_jet(hm_up)
    color_img = Image.fromarray(color, mode="RGB")

    # Blend overlay, weighting the overlay by activation so quiet regions stay
    # close to the underlying anatomy.
    base_arr = np.asarray(base_rgb, dtype=np.float32)
    weight = hm_up[..., None] * alpha
    blended = base_arr * (1 - weight) + color.astype(np.float32) * weight
    overlay_img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")

    return {
        "base": _to_b64_png(base_rgb),
        "heatmap": _to_b64_png(color_img),
        "overlay": _to_b64_png(overlay_img),
    }
