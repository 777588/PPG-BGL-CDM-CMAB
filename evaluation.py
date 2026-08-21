"""Model evaluation with RMSE, MAE, MARD, Clarke error grid, recording-level OOF aggregation, and participant-clustered bootstrap analysis."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def mard(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(y_true == 0):
        raise ValueError("MARD undefined for zero reference value")
    return float(np.mean(np.abs((y_pred - y_true) / y_true)) * 100.0)


def clarke_zone_mgdl(ref: float, pred: float) -> str:
    """Assign Clarke error-grid zones using the standard boundaries implemented in the CBHFF source code."""
    if (ref < 70 and pred < 70) or abs(ref - pred) < 0.2 * ref:
        return "A"
    if ref <= 70 and pred >= 180:
        return "E"
    if ref >= 180 and pred <= 70:
        return "E"
    if ref >= 240 and 70 <= pred <= 180:
        return "D"
    if ref <= 70 <= pred <= 180:
        return "D"
    if 70 <= ref <= 290 and pred >= ref + 110:
        return "C"
    if 130 <= ref <= 180 and pred <= (7.0 / 5.0) * ref - 182:
        return "C"
    return "B"


def clarke_proportions(y_true_mmol, y_pred_mmol) -> Dict[str, float]:
    ref = np.asarray(y_true_mmol, dtype=float) * 18.0
    pred = np.asarray(y_pred_mmol, dtype=float) * 18.0
    zones = np.array([clarke_zone_mgdl(r, p) for r, p in zip(ref, pred)])
    n = len(zones)
    out = {z: float(np.mean(zones == z) * 100.0) for z in "ABCDE"}
    out["A+B"] = out["A"] + out["B"]
    out["zone_labels"] = zones
    return out


def all_metrics(y_true, y_pred) -> Dict[str, float]:
    c = clarke_proportions(y_true, y_pred)
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MARD": mard(y_true, y_pred),
        "Zone_A": c["A"],
        "Zone_AB": c["A+B"],
    }


def segments_to_recording_oof(segment_df: pd.DataFrame) -> pd.DataFrame:
    """Average the two 5-s predictions from each 10-s recording to obtain recording-level OOF predictions."""
    required = {"participant_id", "recording_id", "segment_id", "y_true", "y_pred"}
    missing = required - set(segment_df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    counts = segment_df.groupby(["participant_id", "recording_id"]).size()
    if not np.all(counts.values == 2):
        bad = counts[counts != 2]
        raise ValueError(f"Each recording must have exactly two 5-s predictions. Bad groups:\n{bad}")

    # The two 5-s segments from the same recording must share the same reference glucose value.
    true_nunique = segment_df.groupby(["participant_id", "recording_id"])["y_true"].nunique()
    if not np.all(true_nunique.values == 1):
        raise ValueError("Two segments from a recording have inconsistent reference BGL values")

    rec = (
        segment_df.groupby(["participant_id", "recording_id"], as_index=False)
        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    )
    return rec


def pooled_recording_level_oof(segment_df: pd.DataFrame, expected_recordings: int = 67):
    rec = segments_to_recording_oof(segment_df)
    if len(rec) != expected_recordings:
        print(f"WARNING: pooled recording count is {len(rec)}, expected {expected_recordings} from manuscript")
    metrics = all_metrics(rec["y_true"].to_numpy(), rec["y_pred"].to_numpy())
    return rec, metrics


def participant_clustered_bootstrap_ci(
    recording_df: pd.DataFrame,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, Tuple[float, float]]:
    """Compute a single-model clustered bootstrap confidence interval using participants as the resampling unit."""
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(recording_df["participant_id"].astype(str).unique()))
    point = metric_fn(recording_df["y_true"].to_numpy(), recording_df["y_pred"].to_numpy())
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        blocks = []
        for draw_idx, pid in enumerate(sampled):
            block = recording_df[recording_df["participant_id"].astype(str) == pid].copy()
            # Assign a unique bootstrap-cluster ID to each resampled participant occurrence to preserve multiplicity.
            block["_boot_cluster"] = draw_idx
            blocks.append(block)
        boot = pd.concat(blocks, ignore_index=True)
        vals.append(metric_fn(boot["y_true"].to_numpy(), boot["y_pred"].to_numpy()))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(point), (float(lo), float(hi))


def paired_participant_clustered_bootstrap(
    cmab_recording_df: pd.DataFrame,
    cbhff_recording_df: pd.DataFrame,
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run a paired participant-clustered bootstrap with the difference defined as CMAB minus CBHFF."""
    key = ["participant_id", "recording_id"]
    a = cmab_recording_df.rename(columns={"y_pred": "pred_cmab", "y_true": "true_cmab"})
    b = cbhff_recording_df.rename(columns={"y_pred": "pred_cbhff", "y_true": "true_cbhff"})
    df = a.merge(b, on=key, how="inner")
    if len(df) != len(a) or len(df) != len(b):
        raise ValueError("CMAB and CBHFF recording-level OOF rows are not perfectly matched")
    if not np.allclose(df["true_cmab"], df["true_cbhff"]):
        raise ValueError("Reference BGL differs between paired model predictions")
    df["y_true"] = df["true_cmab"]

    metric_fns = {
        "RMSE": rmse,
        "MAE": mae,
        "MARD": mard,
        "Zone_A": lambda yt, yp: clarke_proportions(yt, yp)["A"],
    }

    def deltas(frame):
        yt = frame["y_true"].to_numpy()
        return {
            name: fn(yt, frame["pred_cmab"].to_numpy()) - fn(yt, frame["pred_cbhff"].to_numpy())
            for name, fn in metric_fns.items()
        }

    point = deltas(df)
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(df["participant_id"].astype(str).unique()))
    boot = {k: [] for k in metric_fns}

    for _ in range(n_boot):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        blocks = []
        for draw_idx, pid in enumerate(sampled):
            block = df[df["participant_id"].astype(str) == pid].copy()
            block["_boot_cluster"] = draw_idx
            blocks.append(block)
        bdf = pd.concat(blocks, ignore_index=True)
        d = deltas(bdf)
        for k, v in d.items():
            boot[k].append(v)

    rows = []
    for k in metric_fns:
        lo, hi = np.quantile(boot[k], [alpha / 2, 1 - alpha / 2])
        rows.append({
            "metric": k,
            "delta_CMAB_minus_CBHFF": point[k],
            "CI_low": float(lo),
            "CI_high": float(hi),
            "CI_excludes_zero": bool(lo > 0 or hi < 0),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    y = np.array([5.0, 6.0, 7.0])
    p = np.array([5.1, 5.8, 7.2])
    print(all_metrics(y, p))
