"""Global precision settings for JAX/NumPy runtime paths."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

JAX_REAL_DTYPE = jnp.float64
JAX_COMPLEX_DTYPE = jnp.complex128
NP_REAL_DTYPE = np.float64
NP_COMPLEX_DTYPE = np.complex128

__all__ = [
    "JAX_REAL_DTYPE",
    "JAX_COMPLEX_DTYPE",
    "NP_REAL_DTYPE",
    "NP_COMPLEX_DTYPE",
]
