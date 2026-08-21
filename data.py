"""Utilities for PPG glucose data loading, label parsing, 10-s cropping, 5-s segmentation, and subject-wise cross-validation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

FS = 2175
CROP_SECONDS = 10
CROP_SAMPLES = FS * CROP_SECONDS
SEGMENT_SECONDS = 5
SEGMENT_SAMPLES = FS * SEGMENT_SECONDS


@dataclass
class LabelSchema:
    """Configure glucose, heart rate, BMI, height, and weight columns in the label CSV; BMI can be calculated from height and weight when missing."""
    glucose_mgdl: Union[str, int] = 3
    heart_rate: Optional[Union[str, int]] = None
    bmi: Optional[Union[str, int]] = None
    height_cm: Optional[Union[str, int]] = None
    weight_kg: Optional[Union[str, int]] = None


@dataclass
class Recording:
    participant_id: str
    recording_id: str
    signal_path: str
    label_path: str
    signal_10s: np.ndarray
    bgl_mmol_l: float
    heart_rate: float
    bmi: float


@dataclass
class Segment:
    participant_id: str
    recording_id: str
    segment_id: int
    signal: np.ndarray
    bgl_mmol_l: float
    heart_rate: float
    bmi: float
    is_synthetic: bool = False
    source: str = "real"


def mgdl_to_mmoll(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.asarray(value) / 18.0 if isinstance(value, np.ndarray) else float(value) / 18.0


def parse_participant_recording_id(path: Union[str, Path]) -> Tuple[str, str]:
    """Parse participant and recording IDs from filenames, prioritizing the expected signal/label naming pattern."""
    stem = Path(path).stem
    m = re.search(r"(?:signal|label)[_-]?(\d+)[_-](\d+)$", stem, flags=re.I)
    if not m:
        nums = re.findall(r"\d+", stem)
        if len(nums) < 2:
            raise ValueError(f"Cannot identify participant/recording ID from: {Path(path).name}")
        pid, rid = nums[-2], nums[-1]
    else:
        pid, rid = m.group(1), m.group(2)
    return pid, rid


def _read_csv_flexible(path: Union[str, Path]) -> pd.DataFrame:
    """Read CSV files with or without a header."""
    path = Path(path)
    try:
        df = pd.read_csv(path)
        # If all column names are numeric, reload without a header to avoid treating the first data row as column names.
        numeric_headers = 0
        for c in df.columns:
            try:
                float(str(c))
                numeric_headers += 1
            except ValueError:
                pass
        if len(df.columns) > 0 and numeric_headers == len(df.columns):
            return pd.read_csv(path, header=None)
        return df
    except Exception:
        return pd.read_csv(path, header=None)


def _col_value(df: pd.DataFrame, spec: Optional[Union[str, int]]) -> Optional[float]:
    if spec is None:
        return None
    if isinstance(spec, int):
        if spec >= df.shape[1]:
            raise IndexError(f"Requested label column {spec}, but file has {df.shape[1]} columns")
        series = pd.to_numeric(df.iloc[:, spec], errors="coerce")
    else:
        if spec not in df.columns:
            raise KeyError(f"Label column '{spec}' not found. Available: {list(df.columns)}")
        series = pd.to_numeric(df[spec], errors="coerce")
    series = series.dropna()
    if len(series) == 0:
        raise ValueError(f"No numeric value found in label column {spec}")
    return float(series.iloc[0])


def read_signal_csv(path: Union[str, Path]) -> np.ndarray:
    """Read a PPG waveform and return it as a one-dimensional float32 array."""
    df = _read_csv_flexible(path)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    # Use the column containing the largest number of valid numeric samples as the PPG signal column.
    counts = numeric.notna().sum(axis=0)
    if counts.max() == 0:
        raise ValueError(f"No numeric PPG samples found in {path}")
    col = counts.idxmax()
    x = numeric[col].dropna().to_numpy(dtype=np.float32)
    return x


def read_label_csv(path: Union[str, Path], schema: LabelSchema) -> Dict[str, float]:
    df = _read_csv_flexible(path)
    glucose = _col_value(df, schema.glucose_mgdl)
    hr = _col_value(df, schema.heart_rate)
    bmi = _col_value(df, schema.bmi)
    height = _col_value(df, schema.height_cm)
    weight = _col_value(df, schema.weight_kg)

    if bmi is None and height is not None and weight is not None:
        bmi = weight / ((height / 100.0) ** 2)
    if hr is None:
        raise ValueError(
            f"Heart-rate column is not configured for {path}. "
            "Set LabelSchema.heart_rate to the actual header or zero-based index."
        )
    if bmi is None:
        raise ValueError(
            f"BMI is unavailable for {path}. Configure LabelSchema.bmi, or height_cm + weight_kg."
        )

    return {
        "bgl_mgdl": float(glucose),
        "bgl_mmol_l": mgdl_to_mmoll(float(glucose)),
        "heart_rate": float(hr),
        "bmi": float(bmi),
    }


def crop_to_10s(x: np.ndarray, fs: int = FS, mode: str = "start") -> np.ndarray:
    """Deterministically crop the raw PPG signal to 10 s from either the start or the center."""
    n = fs * 10
    if len(x) < n:
        raise ValueError(f"Signal has {len(x)} samples; at least {n} are required for a 10-s crop")
    if mode == "start":
        start = 0
    elif mode == "center":
        start = (len(x) - n) // 2
    else:
        raise ValueError("crop mode must be 'start' or 'center'")
    return np.asarray(x[start:start+n], dtype=np.float32).copy()


def split_10s_to_two_5s(x_10s: np.ndarray, fs: int = FS) -> Tuple[np.ndarray, np.ndarray]:
    n5 = fs * 5
    if len(x_10s) != 2 * n5:
        raise ValueError(f"Expected {2*n5} samples, got {len(x_10s)}")
    return x_10s[:n5].copy(), x_10s[n5:].copy()


def _find_label_file(labels_dir: Path, participant_id: str, recording_id: str) -> Path:
    candidates = list(labels_dir.glob(f"*{participant_id}*{recording_id}*.csv"))
    # Prefer label files whose filenames explicitly contain "label".
    candidates.sort(key=lambda p: ("label" not in p.stem.lower(), len(p.name), p.name))
    if not candidates:
        raise FileNotFoundError(
            f"No label CSV found for participant={participant_id}, recording={recording_id}"
        )
    return candidates[0]


def load_recordings(
    raw_data_dir: Union[str, Path],
    labels_dir: Union[str, Path],
    schema: LabelSchema,
    crop_mode: str = "start",
) -> List[Recording]:
    raw_data_dir, labels_dir = Path(raw_data_dir), Path(labels_dir)
    signal_files = sorted(raw_data_dir.glob("*.csv"))
    if not signal_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    recordings: List[Recording] = []
    seen = set()
    for signal_path in signal_files:
        pid, rid = parse_participant_recording_id(signal_path)
        key = (pid, rid)
        if key in seen:
            raise ValueError(f"Duplicate recording key found: {key}")
        seen.add(key)

        label_path = _find_label_file(labels_dir, pid, rid)
        signal = read_signal_csv(signal_path)
        signal_10s = crop_to_10s(signal, mode=crop_mode)
        label = read_label_csv(label_path, schema)
        recordings.append(
            Recording(
                participant_id=pid,
                recording_id=f"{pid}_{rid}",
                signal_path=str(signal_path),
                label_path=str(label_path),
                signal_10s=signal_10s,
                bgl_mmol_l=label["bgl_mmol_l"],
                heart_rate=label["heart_rate"],
                bmi=label["bmi"],
            )
        )
    return recordings


def recordings_to_raw_segments(recordings: Sequence[Recording]) -> List[Segment]:
    segments: List[Segment] = []
    for r in recordings:
        s0, s1 = split_10s_to_two_5s(r.signal_10s)
        for sid, sig in enumerate((s0, s1)):
            segments.append(
                Segment(
                    participant_id=r.participant_id,
                    recording_id=r.recording_id,
                    segment_id=sid,
                    signal=sig,
                    bgl_mmol_l=r.bgl_mmol_l,
                    heart_rate=r.heart_rate,
                    bmi=r.bmi,
                )
            )
    return segments


def make_subjectwise_folds(
    records: Sequence[Union[Recording, Segment]],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[set, set]]:
    """Generate subject sets for five-fold training and testing based on participant IDs."""
    participants = np.array(sorted({r.participant_id for r in records}))
    if len(participants) < n_splits:
        raise ValueError("Number of participants must be >= n_splits")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, te in kf.split(participants):
        folds.append((set(participants[tr]), set(participants[te])))
    return folds


def split_by_participants(items: Sequence, participant_ids: set):
    return [x for x in items if x.participant_id in participant_ids]


def verify_expected_counts(recordings: Sequence[Recording], expected_recordings=67, expected_participants=23):
    n_rec = len(recordings)
    n_pid = len({r.participant_id for r in recordings})
    print(f"Recordings: {n_rec}; participants: {n_pid}")
    if n_rec != expected_recordings or n_pid != expected_participants:
        print(
            "WARNING: counts differ from manuscript dataset "
            f"({expected_recordings} recordings, {expected_participants} participants)."
        )


if __name__ == "__main__":
    # Set the heart-rate and BMI columns according to the actual label CSV before use.
    schema = LabelSchema(glucose_mgdl=3, heart_rate=None, bmi=None)
    print("Edit paths and LabelSchema, then call load_recordings(...).")
