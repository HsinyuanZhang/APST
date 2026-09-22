"""Inference-only association-profile pairing diagnostics."""

from __future__ import annotations
from dataclasses import replace
import numpy as np
from apst.legacy_dandi.carrier import estimate_move_t4
from apst.legacy_dandi.data import SessionData

SEEDS = (20260914, 20260915, 20260916, 20260917, 20260918)


def column_derangement(
    n: int, rng: np.random.Generator, groups: list[np.ndarray] | None = None
) -> np.ndarray:
    """Return source-column indices with no fixed point in each mutable group."""
    permutation = np.arange(n)
    for group in (groups if groups is not None else [np.arange(n)]):
        if len(group) < 2:
            continue
        candidate = group.copy()
        for _ in range(100):
            rng.shuffle(candidate)
            if np.all(candidate != group):
                break
        if np.any(candidate == group):
            candidate = np.roll(group, 1)
        permutation[group] = candidate
    return permutation


def profile_permutation(record: SessionData, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = record.neural.shape[1]
    if kind == "global":
        return column_derangement(n, rng)
    if kind != "rate_matched":
        raise ValueError("kind must be global or rate_matched")
    baseline_rate = estimate_move_t4(record.carrier_counts, record.carrier_angles)[:, 3]
    groups = [
        group
        for group in np.array_split(np.argsort(baseline_rate), 4)
        if len(group) >= 2
    ]
    return column_derangement(n, rng, groups)


def permute_profiles(record: SessionData, kind: str, seed: int) -> SessionData:
    permutation = profile_permutation(record, kind, seed)
    # Neural activity and support stay fixed; only the whole profile columns move.
    return replace(
        record,
        carrier_counts=np.ascontiguousarray(record.carrier_counts[:, permutation]),
    )
