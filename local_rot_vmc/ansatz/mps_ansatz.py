"""Matrix Product State (MPS) ansatz for 1D spin systems."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Union

from netket.utils.types import DType, Array, NNInitFunc
from netket.graph import Lattice


default_kernel_init = nn.initializers.normal(stddev=0.01)


def _create_product_state_tensors(
    L: int,
    D: int,
    symperiod: int,
    config: Array,
    dtype: DType,
) -> Array:
    d = 2
    eye = jnp.eye(D, dtype=dtype)
    zeros = jnp.zeros((D, D), dtype=dtype)

    config_indices = (config == 1).astype(int)

    all_tensors = jnp.zeros((L, d, D, D), dtype=dtype)

    for i in range(L):
        spin_idx = config_indices[i]
        down_tensor = jnp.where(spin_idx == 0, eye, zeros)
        up_tensor = jnp.where(spin_idx == 1, eye, zeros)
        all_tensors = all_tensors.at[i, 0, :, :].set(down_tensor)
        all_tensors = all_tensors.at[i, 1, :, :].set(up_tensor)

    result = []
    for p in range(symperiod):
        period_indices = jnp.array([i for i in range(p, L, symperiod)])
        period_tensors = all_tensors[period_indices]
        avg_tensor = period_tensors.mean(axis=0)
        result.append(avg_tensor)

    return jnp.stack(result, axis=0)


class MPSAnsatz1D(nn.Module):
    r"""Periodic Matrix Product State (MPS) for 1D spin-1/2 systems."""

    lattice: Lattice
    bond_dim: int
    symperiod: Optional[int] = None
    init_config: Union[str, Array, None] = None
    total_sz: Optional[float] = None
    param_dtype: DType = jnp.complex128
    kernel_init: NNInitFunc = default_kernel_init
    noise_scale: float = 1.0

    def setup(self):
        L = self.lattice.extent[0]
        d = 2
        D = self.bond_dim

        self._L, self._d, self._D = L, d, D

        if self.symperiod is None:
            symperiod = L
        else:
            symperiod = self.symperiod

        if L % symperiod != 0 or symperiod <= 0:
            raise ValueError(
                f"System size L={L} must be a multiple of symperiod={symperiod}"
            )
        self._symperiod = symperiod

        if self.init_config is not None:
            if isinstance(self.init_config, str):
                if self.init_config == "ferro":
                    config = jnp.ones(L, dtype=int)
                elif self.init_config == "antiferro" or self.init_config == "neel":
                    config = jnp.array([1 if i % 2 == 0 else -1 for i in range(L)], dtype=int)
                elif self.init_config == "staggered2":
                    config = jnp.array([
                        1 if (i // 2) % 2 == 0 else -1
                        for i in range(L)
                    ], dtype=int)
                elif self.init_config == "antiferro_inv" or self.init_config == "neel_inv":
                    config = jnp.array([-1 if i % 2 == 0 else 1 for i in range(L)], dtype=int)
                elif self.init_config == "positive":
                    def positive_init(key, shape, dtype=self.param_dtype):
                        return jax.random.uniform(
                            key, shape, dtype=dtype,
                            minval=0.0, maxval=self.noise_scale,
                        )
                    self.tensors = self.param(
                        "tensors", positive_init, (symperiod, d, D, D), self.param_dtype
                    )
                    return
                else:
                    raise ValueError(f"Unknown init_config: {self.init_config}")
            else:
                config = jnp.asarray(self.init_config, dtype=int)

            def product_state_init(key, shape, dtype=self.param_dtype):
                base = _create_product_state_tensors(L, D, symperiod, config, dtype)
                noise = self.kernel_init(key, (symperiod, d, D, D), dtype) * self.noise_scale
                return base + noise

            self.tensors = self.param("tensors", product_state_init, (symperiod, d, D, D), self.param_dtype)
        else:
            self.tensors = self.param(
                "tensors", self.kernel_init, (symperiod, d, D, D), self.param_dtype
            )

    def _spin_to_index(self, x: Array) -> Array:
        return (x == 1).astype(int)

    def __call__(self, x: Array) -> Array:
        """Compute log wavefunction amplitude for input spin configurations."""
        psi = self.get_psi_val(x)
        return jnp.log(psi.astype(jnp.complex128) + 1e-100)

    def get_psi_val(self, x: Array) -> Array:
        """Compute raw wavefunction amplitude (no log)."""
        x = jnp.atleast_2d(x)
        qn_x = self._spin_to_index(x)
        psi = jax.vmap(self._contract_mps, in_axes=0)(qn_x)
        if self.total_sz is not None:
            sz = jnp.sum(x, axis=-1) / 2.0
            mask = jnp.isclose(sz, self.total_sz)
            psi = jnp.where(mask, psi, 0.0)
        return psi

    def _contract_mps(self, qn: Array) -> Array:
        all_tensors = jnp.tile(
            self.tensors, (self._L // self._symperiod, 1, 1, 1)
        )

        edge = jnp.eye(self._D, dtype=self.param_dtype)

        for i in range(self._L):
            edge = edge @ all_tensors[i, qn[i], :, :]

        psi = jnp.trace(edge)

        return psi
