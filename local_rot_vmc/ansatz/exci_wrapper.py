"""Generic single-site excitation wrapper."""

from __future__ import annotations

import os

import jax.numpy as jnp
import flax.linen as nn
from netket.utils.types import DType


def splus_single_site_initializer(key, shape, dtype):  # noqa: ARG001
    """Initialize exci_mat as S+ on a single site in basis (↓, ↑)."""
    eps_str = os.environ.get("EXCI_EPS_OVERRIDE", "1e-20")
    eps = float(eps_str)
    return jnp.array(
        [
            [eps, eps],
            [1.0, eps],
        ],
        dtype=dtype,
    )


class ExciWrapper(nn.Module):
    """Wrap any base log-amplitude model with a single-site 2x2 excitation."""

    base: nn.Module
    exci_pos: int = 0
    param_dtype: DType = jnp.complex128

    @nn.compact
    def _compute_psi(self, x):
        N = x.shape[-1]
        pos = self.exci_pos % N

        exci_mat = self.param(
            "exci_mat",
            splus_single_site_initializer,
            (2, 2),
            self.param_dtype,
        )

        bit_i = jnp.where(x[..., pos] == 1, 1, 0)

        x_d = x.at[..., pos].set(-1)
        x_u = x.at[..., pos].set(1)

        log_psi_all = jnp.stack(
            [self.base(x_d), self.base(x_u)], axis=-1,
        )

        is_up = (bit_i == 1)
        coeff_0 = jnp.where(is_up, exci_mat[1, 0], exci_mat[0, 0])
        coeff_1 = jnp.where(is_up, exci_mat[1, 1], exci_mat[0, 1])
        return coeff_0 * jnp.exp(log_psi_all[..., 0]) + coeff_1 * jnp.exp(log_psi_all[..., 1])

    def __call__(self, x):
        psi = self._compute_psi(x)
        eps = jnp.array(1e-300, dtype=self.param_dtype)
        return jnp.log(psi + eps)

    def get_psi_val(self, x):
        """Return ψ(x) directly (no log)."""
        return self._compute_psi(x)
