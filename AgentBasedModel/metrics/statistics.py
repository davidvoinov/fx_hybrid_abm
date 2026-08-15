from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def finite_sample(values: Iterable[float]) -> list[float]:
    out = []
    for value in values:
        if isinstance(value, bool):
            value = float(value)
        if value is None:
            continue
        if math.isfinite(value):
            out.append(float(value))
    return out


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 2000,
                      ci_level: float = 0.95, seed: int | None = None) -> dict:
    sample = np.asarray(finite_sample(values), dtype=float)
    n = int(sample.size)
    if n == 0:
        return {
            'n': 0,
            'mean': float('nan'),
            'std': float('nan'),
            'median': float('nan'),
            'ci_lo': float('nan'),
            'ci_hi': float('nan'),
        }

    mean_value = float(sample.mean())
    std_value = float(sample.std(ddof=1)) if n > 1 else 0.0
    median_value = float(np.median(sample))
    if n_boot <= 0 or n == 1:
        return {
            'n': n,
            'mean': mean_value,
            'std': std_value,
            'median': median_value,
            'ci_lo': mean_value,
            'ci_hi': mean_value,
        }

    alpha = max(0.0, min(1.0, 1.0 - ci_level)) / 2.0
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_means = sample[sample_idx].mean(axis=1)
    ci_lo, ci_hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return {
        'n': n,
        'mean': mean_value,
        'std': std_value,
        'median': median_value,
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
    }


def independent_permutation_test(sample_a: Sequence[float], sample_b: Sequence[float],
                                 n_perm: int = 10000,
                                 seed: int | None = None) -> dict:
    clean_a = finite_sample(sample_a)
    clean_b = finite_sample(sample_b)
    mean_a = float(np.mean(clean_a)) if clean_a else float('nan')
    mean_b = float(np.mean(clean_b)) if clean_b else float('nan')
    mean_diff = mean_a - mean_b if math.isfinite(mean_a) and math.isfinite(mean_b) else float('nan')

    if len(clean_a) < 2 or len(clean_b) < 2:
        return {
            'n_a': len(clean_a),
            'n_b': len(clean_b),
            'mean_a': mean_a,
            'mean_b': mean_b,
            'mean_diff': mean_diff,
            'stat': float('nan'),
            'p_value': float('nan'),
        }

    clean_a_arr = np.asarray(clean_a, dtype=float)
    clean_b_arr = np.asarray(clean_b, dtype=float)
    n_a = len(clean_a)
    n_b = len(clean_b)
    observed_abs = abs(mean_diff)

    if n_a == 1 and n_b == 1:
        return {
            'n_a': n_a,
            'n_b': n_b,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'mean_diff': mean_diff,
            'stat': mean_diff,
            'p_value': 1.0,
        }

    pooled = np.concatenate([clean_a_arr, clean_b_arr])
    n_perm = max(1, int(n_perm))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        permuted = rng.permutation(pooled)
        perm_diff = float(permuted[:n_a].mean() - permuted[n_a:].mean())
        if abs(perm_diff) >= observed_abs:
            exceed += 1

    return {
        'n_a': n_a,
        'n_b': n_b,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'mean_diff': mean_diff,
        'stat': mean_diff,
        'p_value': float((exceed + 1) / (n_perm + 1)),
    }


def bootstrap_diff_ci(sample_a: Sequence[float], sample_b: Sequence[float],
                      n_boot: int = 2000, ci_level: float = 0.95,
                      seed: int | None = None) -> dict:
    clean_a = np.asarray(finite_sample(sample_a), dtype=float)
    clean_b = np.asarray(finite_sample(sample_b), dtype=float)
    n_a = int(clean_a.size)
    n_b = int(clean_b.size)
    mean_a = float(clean_a.mean()) if n_a else float('nan')
    mean_b = float(clean_b.mean()) if n_b else float('nan')
    mean_diff = mean_a - mean_b if math.isfinite(mean_a) and math.isfinite(mean_b) else float('nan')

    if n_a == 0 or n_b == 0:
        return {
            'n_a': n_a,
            'n_b': n_b,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'mean_diff': mean_diff,
            'ci_lo': float('nan'),
            'ci_hi': float('nan'),
        }

    if n_boot <= 0 or n_a == 1 or n_b == 1:
        return {
            'n_a': n_a,
            'n_b': n_b,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'mean_diff': mean_diff,
            'ci_lo': mean_diff,
            'ci_hi': mean_diff,
        }

    alpha = max(0.0, min(1.0, 1.0 - ci_level)) / 2.0
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, n_a, size=(int(n_boot), n_a))
    idx_b = rng.integers(0, n_b, size=(int(n_boot), n_b))
    boot_diff = clean_a[idx_a].mean(axis=1) - clean_b[idx_b].mean(axis=1)
    ci_lo, ci_hi = np.quantile(boot_diff, [alpha, 1.0 - alpha])
    return {
        'n_a': n_a,
        'n_b': n_b,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'mean_diff': mean_diff,
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
    }


def paired_permutation_test(sample_a: Sequence[float], sample_b: Sequence[float],
                            n_perm: int = 10000,
                            seed: int | None = None) -> dict:
    clean_pairs = [
        (float(value_a), float(value_b))
        for value_a, value_b in zip(sample_a, sample_b)
        if value_a is not None and value_b is not None
        and math.isfinite(value_a) and math.isfinite(value_b)
    ]
    n = len(clean_pairs)
    if n == 0:
        return {
            'n': 0,
            'mean_a': float('nan'),
            'mean_b': float('nan'),
            'mean_diff': float('nan'),
            'stat': float('nan'),
            'p_value': float('nan'),
        }

    clean_a = np.asarray([pair[0] for pair in clean_pairs], dtype=float)
    clean_b = np.asarray([pair[1] for pair in clean_pairs], dtype=float)
    diff = clean_a - clean_b
    mean_a = float(clean_a.mean())
    mean_b = float(clean_b.mean())
    mean_diff = float(diff.mean())

    if n == 1:
        return {
            'n': n,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'mean_diff': mean_diff,
            'stat': mean_diff,
            'p_value': 1.0,
        }

    n_perm = max(1, int(n_perm))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n), replace=True)
    perm_stats = (signs * diff).mean(axis=1)
    observed_abs = abs(mean_diff)
    exceed = int(np.count_nonzero(np.abs(perm_stats) >= observed_abs))
    p_value = float((exceed + 1) / (n_perm + 1))

    return {
        'n': n,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'mean_diff': mean_diff,
        'stat': mean_diff,
        'p_value': p_value,
    }


def bootstrap_paired_diff_ci(sample_a: Sequence[float], sample_b: Sequence[float],
                             n_boot: int = 2000, ci_level: float = 0.95,
                             seed: int | None = None) -> dict:
    clean_pairs = [
        (float(value_a), float(value_b))
        for value_a, value_b in zip(sample_a, sample_b)
        if value_a is not None and value_b is not None
        and math.isfinite(value_a) and math.isfinite(value_b)
    ]
    n = len(clean_pairs)
    if n == 0:
        return {
            'n': 0,
            'mean_a': float('nan'),
            'mean_b': float('nan'),
            'mean_diff': float('nan'),
            'ci_lo': float('nan'),
            'ci_hi': float('nan'),
        }

    clean_a = np.asarray([pair[0] for pair in clean_pairs], dtype=float)
    clean_b = np.asarray([pair[1] for pair in clean_pairs], dtype=float)
    diff = clean_a - clean_b
    mean_a = float(clean_a.mean())
    mean_b = float(clean_b.mean())
    mean_diff = float(diff.mean())

    if n_boot <= 0 or n == 1:
        return {
            'n': n,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'mean_diff': mean_diff,
            'ci_lo': mean_diff,
            'ci_hi': mean_diff,
        }

    alpha = max(0.0, min(1.0, 1.0 - ci_level)) / 2.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_diff = diff[idx].mean(axis=1)
    ci_lo, ci_hi = np.quantile(boot_diff, [alpha, 1.0 - alpha])
    return {
        'n': n,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'mean_diff': mean_diff,
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
    }