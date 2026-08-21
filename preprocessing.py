"""PPG preprocessing: fourth-order 0.30-15 Hz Butterworth band-pass filtering, SOS zero-phase filtering, 5-s segmentation, and segment-wise z-score normalization."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 2175
LOWCUT = 0.30
HIGHCUT = 15.00
ORDER = 4


def design_bandpass_sos(
    fs: int = FS,
    lowcut: float = LOWCUT,
    highcut: float = HIGHCUT,
    order: int = ORDER,
) -> np.ndarray:
    return butter(
        N=order,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )


def butterworth_filter_10s(x_10s: np.ndarray, sos: np.ndarray | None = None) -> np.ndarray:
    if sos is None:
        sos = design_bandpass_sos()
    # Keep SciPy default edge-padding settings by leaving padtype and padlen unspecified.
    return sosfiltfilt(sos, np.asarray(x_10s, dtype=np.float64)).astype(np.float32)


def zscore_segment(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = float(x.mean())
    std = float(x.std(ddof=0))
    if std < eps:
        raise ValueError("Near-constant segment cannot be z-score normalized safely")
    return ((x - mean) / std).astype(np.float32)


def preprocess_10s_recording(x_10s: np.ndarray, fs: int = FS) -> Tuple[np.ndarray, np.ndarray]:
    """Apply preprocessing in the order: filter the 10-s recording, split into two 5-s segments, then normalize each segment independently."""
    filtered = butterworth_filter_10s(x_10s)
    n5 = fs * 5
    if len(filtered) != 2 * n5:
        raise ValueError(f"Expected {2*n5} samples after 10-s crop, got {len(filtered)}")
    s0 = zscore_segment(filtered[:n5])
    s1 = zscore_segment(filtered[n5:])
    return s0, s1


def preprocess_recordings(recordings: Sequence) -> List:
    """Convert recording objects into 5-s segment dictionaries containing signals, labels, and metadata."""
    out = []
    for r in recordings:
        s0, s1 = preprocess_10s_recording(r.signal_10s)
        for sid, sig in enumerate((s0, s1)):
            out.append({
                "participant_id": r.participant_id,
                "recording_id": r.recording_id,
                "segment_id": sid,
                "signal": sig,
                "bgl_mmol_l": float(r.bgl_mmol_l),
                "heart_rate": float(r.heart_rate),
                "bmi": float(r.bmi),
                "is_synthetic": False,
                "source": "real",
            })
    return out
