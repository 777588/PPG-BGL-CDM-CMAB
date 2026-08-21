"""MFO hyperparameter search that independently optimizes the learning rate, BiLSTM hidden sizes, dropout rate, and attention coefficient within each outer fold."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple
import math

import numpy as np


@dataclass
class CMABParams:
    lr: float
    h1: int
    h2: int
    h3: int
    dropout: float
    lambda_att: float


class CMABSearchSpace:
    """The internal search vector represents log10 learning rate, three BiLSTM hidden sizes, dropout rate, and attention coefficient."""
    lower = np.array([-4.0, 4.0, 8.0, 8.0, 0.2, 0.1], dtype=float)
    upper = np.array([-2.0, 16.0, 20.0, 24.0, 0.5, 0.8], dtype=float)

    @classmethod
    def clip(cls, x: np.ndarray) -> np.ndarray:
        return np.clip(x, cls.lower, cls.upper)

    @classmethod
    def decode(cls, x: np.ndarray) -> CMABParams:
        x = cls.clip(np.asarray(x, dtype=float).copy())
        h1 = int(np.clip(np.rint(x[1]), 4, 16))
        h2 = int(np.clip(np.rint(x[2]), 8, 20))
        h3 = int(np.clip(np.rint(x[3]), 8, 24))
        return CMABParams(
            lr=float(10.0 ** x[0]),
            h1=h1,
            h2=h2,
            h3=h3,
            dropout=float(np.clip(x[4], 0.2, 0.5)),
            lambda_att=float(np.clip(x[5], 0.1, 0.8)),
        )


class MothFlameOptimizer:
    def __init__(
        self,
        fitness_fn: Callable[[CMABParams], float],
        n_moths: int = 10,
        max_iter: int = 20,
        b: float = 1.0,
        patience: int = 5,
        seed: int = 42,
    ):
        self.fitness_fn = fitness_fn
        self.n_moths = n_moths
        self.max_iter = max_iter
        self.b = b
        self.patience = patience
        self.rng = np.random.default_rng(seed)
        self.evaluations = 0

    def _evaluate(self, positions: np.ndarray) -> np.ndarray:
        fitness = np.empty(len(positions), dtype=float)
        for i, p in enumerate(positions):
            fitness[i] = float(self.fitness_fn(CMABSearchSpace.decode(p)))
            self.evaluations += 1
        return fitness

    def optimize(self) -> Tuple[CMABParams, float, List[Dict]]:
        lo, hi = CMABSearchSpace.lower, CMABSearchSpace.upper
        moths = self.rng.uniform(lo, hi, size=(self.n_moths, len(lo)))

        fitness = self._evaluate(moths)
        order = np.argsort(fitness)
        flames = moths[order].copy()
        flame_fit = fitness[order].copy()

        best_pos = flames[0].copy()
        best_fit = float(flame_fit[0])
        no_improve = 0
        history: List[Dict] = []

        for iteration in range(1, self.max_iter + 1):
            # Gradually reduce the number of active flames according to the MFO update rule.
            flame_no = int(round(self.n_moths - iteration * ((self.n_moths - 1) / self.max_iter)))
            flame_no = max(1, min(self.n_moths, flame_no))

            # Decrease parameter a approximately linearly from -1 to -2 to control the logarithmic spiral search.
            a = -1.0 - iteration / self.max_iter
            new_moths = np.empty_like(moths)
            for i in range(self.n_moths):
                flame_idx = i if i < flame_no else flame_no - 1
                flame = flames[flame_idx]
                distance = np.abs(flame - moths[i])
                t = (a - 1.0) * self.rng.random(len(lo)) + 1.0
                new_moths[i] = distance * np.exp(self.b * t) * np.cos(2 * np.pi * t) + flame

            moths = np.clip(new_moths, lo, hi)
            fitness = self._evaluate(moths)

            # Merge previous flames with updated moths and retain the N best positions as flames for the next iteration.
            all_pos = np.vstack([flames, moths])
            all_fit = np.concatenate([flame_fit, fitness])
            order = np.argsort(all_fit)[: self.n_moths]
            flames = all_pos[order].copy()
            flame_fit = all_fit[order].copy()

            current_best = float(flame_fit[0])
            if current_best < best_fit - 1e-12:
                best_fit = current_best
                best_pos = flames[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            history.append({
                "iteration": iteration,
                "active_flames": flame_no,
                "best_rmse": best_fit,
                "evaluations": self.evaluations,
                **asdict(CMABSearchSpace.decode(best_pos)),
            })
            print(
                f"MFO iter {iteration:02d}: best val RMSE={best_fit:.6f}, "
                f"active flames={flame_no}, evaluations={self.evaluations}"
            )

            if no_improve >= self.patience:
                break

        return CMABSearchSpace.decode(best_pos), best_fit, history


def run_independent_outer_fold_searches(
    fitness_factory: Callable[[int], Callable[[CMABParams], float]],
    n_outer_folds: int = 5,
    seed: int = 42,
):
    """Reinitialize and run MFO independently for each outer fold without reusing a global hyperparameter set."""
    results = {}
    for fold in range(1, n_outer_folds + 1):
        print(f"\n=== Outer Fold {fold} MFO ===")
        opt = MothFlameOptimizer(
            fitness_fn=fitness_factory(fold),
            n_moths=10,
            max_iter=20,
            b=1.0,
            patience=5,
            seed=seed,
        )
        best_params, best_rmse, hist = opt.optimize()
        results[fold] = {
            "best_params": asdict(best_params),
            "best_val_rmse": best_rmse,
            "evaluations": opt.evaluations,
            "history": hist,
        }
    return results
