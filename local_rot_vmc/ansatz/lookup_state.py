"""Exact lookup-table ansatz on a fixed-(N, total_sz) basis."""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def _encode_total_sz(total_sz: float) -> int:
    return int(round(2.0 * float(total_sz)))


def _decode_total_sz(code: int) -> float:
    return float(code) / 2.0


def _fixed_sz_basis_states(N: int, total_sz: float) -> np.ndarray:
    n_up = int(round(N / 2 + float(total_sz)))
    basis = np.full((int(math.comb(N, n_up)), N), -1, dtype=np.int8)
    for row_idx, up_sites in enumerate(combinations(range(N), n_up)):
        basis[row_idx, list(up_sites)] = 1
    return basis[::-1].copy()


def _spin_configs_to_keys(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.int8)
    if states.ndim != 2:
        raise ValueError(f"states must have shape (n_states, N), got {states.shape}.")
    N = states.shape[1]
    weights = (np.uint64(1) << np.arange(N, dtype=np.uint64))
    return np.sum((states == 1).astype(np.uint64) * weights[None, :], axis=1, dtype=np.uint64)


@lru_cache(maxsize=None)
def _lookup_constants(N: int, total_sz_code: int):
    total_sz = _decode_total_sz(total_sz_code)
    basis_states = _fixed_sz_basis_states(int(N), float(total_sz))
    keys = _spin_configs_to_keys(basis_states)
    order_search = np.argsort(keys)
    keys_sorted = np.asarray(keys[order_search], dtype=np.uint64)
    indices_sorted = np.asarray(order_search, dtype=np.int32)
    weights = np.asarray((np.uint64(1) << np.arange(int(N), dtype=np.uint64)), dtype=np.uint64)
    return keys_sorted, indices_sorted, weights


class LookupState(nn.Module):
    """Flax ansatz that returns log-amplitudes from a fixed-Sz amplitude table."""

    N: int
    total_sz: float
    param_dtype: type = jnp.complex128
    init_stddev: float = 0.5
    zero_amplitude_floor: float = 1e-13

    @nn.compact
    def __call__(self, sigma):
        sigma = jnp.asarray(sigma, dtype=jnp.int8)
        if sigma.ndim == 1:
            sigma = sigma[None, :]

        total_sz_code = _encode_total_sz(self.total_sz)
        keys_sorted_np, indices_sorted_np, weights_np = _lookup_constants(int(self.N), total_sz_code)
        keys_sorted = jnp.asarray(keys_sorted_np, dtype=jnp.uint64)
        indices_sorted = jnp.asarray(indices_sorted_np, dtype=jnp.int32)
        weights = jnp.asarray(weights_np, dtype=jnp.uint64)
        n_basis = int(indices_sorted.shape[0])
        amplitudes = self.param(
            "amplitudes",
            lambda key, shape, dtype: (
                self.init_stddev * jax.random.normal(key, shape, dtype=dtype)
            ),
            (n_basis,),
            self.param_dtype,
        )

        bits = (sigma == 1).astype(jnp.uint64)
        query_keys = jnp.sum(bits * weights[None, :], axis=1, dtype=jnp.uint64)
        pos = jnp.searchsorted(keys_sorted, query_keys, side="left")
        pos = jnp.clip(pos, 0, keys_sorted.shape[0] - 1)
        found = keys_sorted[pos] == query_keys
        basis_indices = indices_sorted[pos]
        values = amplitudes[basis_indices]
        floor = jnp.asarray(self.zero_amplitude_floor, dtype=values.real.dtype)
        safe_values = jnp.where(jnp.abs(values) > floor, values, floor + 0.0j)
        log_values = jnp.log(safe_values)
        return jnp.where(found, log_values, jnp.full_like(log_values, -jnp.inf + 0.0j))

    @nn.compact
    def get_psi_val(self, sigma):
        """Return raw amplitude psi(sigma) without log-floor, preserving gradients at zero."""
        sigma = jnp.asarray(sigma, dtype=jnp.int8)
        if sigma.ndim == 1:
            sigma = sigma[None, :]

        total_sz_code = _encode_total_sz(self.total_sz)
        keys_sorted_np, indices_sorted_np, weights_np = _lookup_constants(int(self.N), total_sz_code)
        keys_sorted = jnp.asarray(keys_sorted_np, dtype=jnp.uint64)
        indices_sorted = jnp.asarray(indices_sorted_np, dtype=jnp.int32)
        weights = jnp.asarray(weights_np, dtype=jnp.uint64)
        n_basis = int(indices_sorted.shape[0])
        amplitudes = self.param(
            "amplitudes",
            lambda key, shape, dtype: (
                self.init_stddev * jax.random.normal(key, shape, dtype=dtype)
            ),
            (n_basis,),
            self.param_dtype,
        )

        bits = (sigma == 1).astype(jnp.uint64)
        query_keys = jnp.sum(bits * weights[None, :], axis=1, dtype=jnp.uint64)
        pos = jnp.searchsorted(keys_sorted, query_keys, side="left")
        pos = jnp.clip(pos, 0, keys_sorted.shape[0] - 1)
        found = keys_sorted[pos] == query_keys
        basis_indices = indices_sorted[pos]
        values = amplitudes[basis_indices]
        return jnp.where(found, values, jnp.zeros_like(values))
