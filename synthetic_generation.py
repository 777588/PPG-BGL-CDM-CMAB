"""CDM synthetic-data generation using only real segments from the outer-training partition; generate five candidates per sample and randomly retain two."""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import torch


def _conditions_array(segments: Sequence[Dict]) -> np.ndarray:
    return np.asarray(
        [[s["bgl_mmol_l"], s["heart_rate"], s["bmi"]] for s in segments],
        dtype=np.float32,
    )


@torch.no_grad()
def generate_five_retain_two_cdm(
    model,
    diffusion,
    condition_scaler,
    real_train_segments: Sequence[Dict],
    outer_test_participant_ids: Optional[Set[str]] = None,
    seed: int = 42,
    n_candidates: int = 5,
    n_retain: int = 2,
    sample_batch_size: int = 8,
) -> List[Dict]:
    if n_retain > n_candidates:
        raise ValueError("n_retain cannot exceed n_candidates")
    if any(bool(s.get("is_synthetic", False)) for s in real_train_segments):
        raise ValueError("CDM generation must start from REAL training segments only")

    train_pids = {str(s["participant_id"]) for s in real_train_segments}
    if outer_test_participant_ids is not None:
        leakage = train_pids.intersection({str(x) for x in outer_test_participant_ids})
        if leakage:
            raise RuntimeError(f"Outer-test leakage detected in generation input: {sorted(leakage)}")

    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(seed)

    conditions = _conditions_array(real_train_segments)
    cond_std = condition_scaler.transform(conditions)

    # Repeat each real sample condition n_candidates times for batched candidate generation.
    repeated = np.repeat(cond_std, n_candidates, axis=0)
    synth_chunks = []
    for start in range(0, len(repeated), sample_batch_size):
        c = torch.tensor(repeated[start:start+sample_batch_size], dtype=torch.float32, device=device)
        x = diffusion.sample(model, c, length=10875, generator=torch_gen)
        synth_chunks.append(x.squeeze(1).cpu().numpy().astype(np.float32))
    synth = np.concatenate(synth_chunks, axis=0).reshape(len(real_train_segments), n_candidates, -1)

    output: List[Dict] = [deepcopy(s) for s in real_train_segments]
    for i, s in enumerate(real_train_segments):
        keep = rng.choice(n_candidates, size=n_retain, replace=False)
        for j, k in enumerate(keep):
            item = deepcopy(s)
            item["signal"] = synth[i, int(k)]
            item["is_synthetic"] = True
            item["source"] = "cdm"
            item["synthetic_id"] = int(j)
            item["candidate_index"] = int(k)
            output.append(item)
    return output


def assert_outer_test_is_real_only(test_segments: Sequence[Dict]):
    bad = [s for s in test_segments if bool(s.get("is_synthetic", False))]
    if bad:
        raise RuntimeError("Outer-test fold must contain only real PPG segments")
