"""Conventional augmentation using time warping, amplitude scaling, and Gaussian noise; generate five candidates per real segment and randomly retain two."""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.interpolate import CubicSpline


def time_warp(
    x: np.ndarray,
    rng: np.random.Generator,
    n_knots: int = 5,
    sigma: float = 0.02,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    t = np.linspace(0.0, 1.0, n)
    knots = np.linspace(0.0, 1.0, n_knots)

    # Apply random temporal displacements to the internal control points while keeping both endpoints fixed.
    disp = rng.normal(0.0, sigma, size=n_knots)
    disp[0] = 0.0
    disp[-1] = 0.0
    warped_knots = knots + disp
    warped_knots[0] = 0.0
    warped_knots[-1] = 1.0

    # Enforce a strictly monotonic time mapping to prevent temporal order reversal during interpolation.
    eps = 1e-4
    for i in range(1, n_knots - 1):
        lo = warped_knots[i - 1] + eps
        hi = 1.0 - (n_knots - 1 - i) * eps
        warped_knots[i] = np.clip(warped_knots[i], lo, hi)
    for i in range(n_knots - 2, 0, -1):
        warped_knots[i] = min(warped_knots[i], warped_knots[i + 1] - eps)

    warp_curve = CubicSpline(knots, warped_knots, bc_type="natural")(t)
    warp_curve = np.clip(warp_curve, 0.0, 1.0)
    warp_curve = np.maximum.accumulate(warp_curve)
    warp_curve[-1] = 1.0

    # Resample the original waveform on the warped time coordinates.
    y = np.interp(warp_curve, t, x)
    return y.astype(np.float32)


def scaling(x: np.ndarray, rng: np.random.Generator, low=0.95, high=1.05) -> np.ndarray:
    factor = rng.uniform(low, high)
    return (np.asarray(x, dtype=np.float32) * factor).astype(np.float32)


def gaussian_noise(x: np.ndarray, rng: np.random.Generator, ratio: float = 0.02) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    sigma = ratio * float(x.std(ddof=0))
    return (x + rng.normal(0.0, sigma, size=x.shape)).astype(np.float32)


def augment_one(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = time_warp(x, rng=rng, n_knots=5, sigma=0.02)
    y = scaling(y, rng=rng, low=0.95, high=1.05)
    y = gaussian_noise(y, rng=rng, ratio=0.02)
    return y


def generate_five_retain_two(
    real_segments: Sequence[Dict],
    seed: int = 42,
    n_candidates: int = 5,
    n_retain: int = 2,
) -> List[Dict]:
    """Return the real segments together with two randomly retained augmented samples for each real segment."""
    if n_retain > n_candidates:
        raise ValueError("n_retain cannot exceed n_candidates")
    rng = np.random.default_rng(seed)
    output: List[Dict] = [deepcopy(s) for s in real_segments]

    for s in real_segments:
        candidates = [augment_one(s["signal"], rng) for _ in range(n_candidates)]
        keep = rng.choice(n_candidates, size=n_retain, replace=False)
        for j, idx in enumerate(keep):
            item = deepcopy(s)
            item["signal"] = candidates[int(idx)]
            item["is_synthetic"] = True
            item["source"] = "traditional"
            item["augmentation_id"] = int(j)
            output.append(item)
    return output
