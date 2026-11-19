"""Random initial spin configurations for MCMC chains."""

import numpy as np
from typing import List, Optional, Tuple


def generate_spin_configs(
    N: int,
    total_sz: float,
    fixed_up_sites: Optional[List[int]] = None,
    n_configs: Optional[int] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    if fixed_up_sites is None:
        fixed_up_sites = []
    if n_configs is None:
        n_configs = 1

    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if n_configs <= 0:
        raise ValueError(f"n_configs must be positive, got {n_configs}")

    if any(site < 0 or site >= N for site in fixed_up_sites):
        raise ValueError(f"Site indices must be in [0, {N-1}]")

    n_up_float = N / 2 + total_sz
    n_up_required = int(round(n_up_float))
    if not np.isclose(n_up_float, n_up_required):
        raise ValueError(
            f"Invalid total_sz={total_sz} for N={N}: "
            "N/2 + total_sz must be an integer"
        )
    n_down_required = N - n_up_required

    if n_up_required < 0 or n_up_required > N:
        raise ValueError(
            f"Cannot achieve total_sz={total_sz} with N={N} sites. "
            f"Required n_up={n_up_required}, n_down={n_down_required}"
        )

    n_fixed_up = len(fixed_up_sites)
    if n_fixed_up > n_up_required:
        raise ValueError(
            f"Cannot satisfy constraints: {n_fixed_up} sites must be up, "
            f"but total_sz={total_sz} only allows {n_up_required} up spins"
        )

    n_up_remaining = n_up_required - n_fixed_up

    fixed_up_sites = sorted(set(fixed_up_sites))
    free_sites = np.setdiff1d(
        np.arange(N, dtype=np.int64),
        np.array(fixed_up_sites, dtype=np.int64),
        assume_unique=True
    )

    if n_up_remaining > len(free_sites):
        raise ValueError(
            f"Cannot satisfy constraints: need {n_up_remaining} more up spins "
            f"but only {len(free_sites)} free sites available"
        )

    configs = np.full((n_configs, N), -1, dtype=np.int8)
    if fixed_up_sites:
        configs[:, fixed_up_sites] = 1

    rng = np.random.default_rng(seed)
    if n_up_remaining > 0:
        for i in range(n_configs):
            up_choices = rng.choice(free_sites, size=n_up_remaining, replace=False)
            configs[i, up_choices] = 1

    return configs
