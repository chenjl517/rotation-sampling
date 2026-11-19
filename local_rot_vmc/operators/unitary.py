"""Unitary matrix operations for local-rotation VMC."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, List, Dict


def random_unitary_block(
    key: jax.random.PRNGKey,
    dim: int = 4,
    dtype: jnp.dtype = jnp.complex128
) -> jnp.ndarray:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        key1, key2 = jax.random.split(key)
        A_real = jax.random.normal(key1, (dim, dim))
        A_imag = jax.random.normal(key2, (dim, dim))
        A = A_real + 1j * A_imag
    else:
        A = jax.random.normal(key, (dim, dim))
    Q, _ = jnp.linalg.qr(A)
    return Q.astype(dtype)

def _generate_sz_conserving_unitary(key, dtype=jnp.complex128):
    keys = jax.random.split(key, 3)

    phi1 = jnp.exp(1j * 2 * jnp.pi * jax.random.uniform(keys[0]))
    phi2 = jnp.exp(1j * 2 * jnp.pi * jax.random.uniform(keys[2]))

    U1 = jnp.array([[phi1]], dtype=dtype)
    U2 = random_unitary_block(keys[1], 2, dtype=dtype)
    U3 = jnp.array([[phi2]], dtype=dtype)

    zero12 = jnp.zeros((1, 2), dtype=dtype)
    zero21 = jnp.zeros((2, 1), dtype=dtype)
    zero11 = jnp.zeros((1, 1), dtype=dtype)

    return jnp.block([
        [U1,      zero12, zero11],
        [zero21,  U2,     zero21],
        [zero11,  zero12, U3],
    ]).astype(dtype)


def _generate_block2x2_unitary(key, dtype=jnp.complex128):
    key_a, key_b = jax.random.split(key)
    block_a = random_unitary_block(key_a, 2, dtype=dtype)
    block_b = random_unitary_block(key_b, 2, dtype=dtype)

    U = jnp.zeros((4, 4), dtype=dtype)
    U = U.at[0, 0].set(block_a[0, 0])
    U = U.at[0, 3].set(block_a[0, 1])
    U = U.at[3, 0].set(block_a[1, 0])
    U = U.at[3, 3].set(block_a[1, 1])
    U = U.at[1, 1].set(block_b[0, 0])
    U = U.at[1, 2].set(block_b[0, 1])
    U = U.at[2, 1].set(block_b[1, 0])
    U = U.at[2, 2].set(block_b[1, 1])
    return U


def generate_random_unitary(key, dim=4, dtype=jnp.complex128, *, scheme="general", conserve_sz=None):
    """Generate a random two-spin unitary according to a named scheme."""
    if conserve_sz is not None:
        if isinstance(conserve_sz, (bool, np.bool_)):
            scheme = "sz_conserving" if bool(conserve_sz) else "general"
        else:
            raise TypeError(
                "generate_random_unitary(..., conserve_sz=...) only accepts bool for backward "
                "compatibility. Use scheme='general', 'sz_conserving', or 'block2x2'."
            )

    if scheme == "general":
        return random_unitary_block(key, dim, dtype=dtype)

    if dim != 4:
        raise ValueError(f"scheme='{scheme}' requires dim=4, got dim={dim}.")

    if scheme == "sz_conserving":
        return _generate_sz_conserving_unitary(key, dtype=dtype)

    if scheme == "block2x2":
        return _generate_block2x2_unitary(key, dtype=dtype)

    raise ValueError(f"Unknown unitary scheme: {scheme!r}")


def generate_hadamard_unitary(
    dim: int = 4,
    dtype: jnp.dtype = jnp.complex128
) -> jnp.ndarray:
    """Fixed Hadamard rotation."""
    H2 = jnp.array([[1, 1], [1, -1]], dtype=dtype) / jnp.sqrt(2.0)
    if dim == 2:
        return H2
    if dim == 4:
        U = jnp.eye(4, dtype=dtype)
        return U.at[1:3, 1:3].set(H2)
    raise NotImplementedError("Hadamard unitary is only implemented for dim=2 and dim=4.")


def generate_identity_unitary(
    dim: int = 4,
    dtype: jnp.dtype = jnp.complex128
) -> jnp.ndarray:
    """Generate identity unitary matrix (no rotation)."""
    return jnp.eye(dim, dtype=dtype)


class UnitaryGenerator:
    """Strategy pattern for generating unitary matrices."""

    def __init__(self, strategy: str = "fixed", dim: int = 4, dtype=jnp.complex128):
        """Initialize unitary generator."""
        if strategy not in ["fixed", "random", "identity"]:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'fixed', 'random', or 'identity'.")

        self.strategy = strategy
        self.dim = dim
        self.dtype = dtype
        self.key = jax.random.PRNGKey(42)

    def generate(self, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """Generate a unitary matrix based on the strategy."""
        if self.strategy == "fixed":
            return generate_hadamard_unitary(self.dim, self.dtype)

        elif self.strategy == "identity":
            return generate_identity_unitary(self.dim, self.dtype)

        elif self.strategy == "random":
            if key is None:
                self.key, key = jax.random.split(self.key)
            return generate_random_unitary(key, dim=self.dim, dtype=self.dtype, scheme="general")

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


def two_spin_config_to_index(spins: jnp.ndarray, i: int, j: int) -> int:
    """Convert two spins (spins[i], spins[j]) to index 0-3."""
    bit_i = jnp.where(spins[i] == 1, 0, 1)
    bit_j = jnp.where(spins[j] == 1, 0, 1)
    return bit_i * 2 + bit_j


def index_to_two_spin_config(index: int, i: int, j: int, N: int) -> jnp.ndarray:
    """Convert index 0-3 to two spins at positions i and j."""
    raise NotImplementedError("Use apply_two_spin_config instead")


def apply_two_spin_config(
    spins: jnp.ndarray,
    i: int,
    j: int,
    config_index: int
) -> jnp.ndarray:
    """Apply a 2-bit configuration (0-3) to spins at positions i and j."""
    bit_i = (config_index >> 1) & 1
    bit_j = config_index & 1

    spin_i = jnp.where(bit_i == 0, 1, -1)
    spin_j = jnp.where(bit_j == 0, 1, -1)

    spins_new = spins.at[i].set(spin_i).at[j].set(spin_j)
    return spins_new


def batch_two_spin_config_to_index(
    spins_batch: jnp.ndarray,
    i: int,
    j: int
) -> jnp.ndarray:
    """Batch version of two_spin_config_to_index."""
    bit_i = jnp.where(spins_batch[:, i] == 1, 0, 1)
    bit_j = jnp.where(spins_batch[:, j] == 1, 0, 1)
    return bit_i * 2 + bit_j


def spins_to_index(spins: jnp.ndarray) -> int:
    """Convert full spin configuration to integer index."""
    bits = jnp.where(spins == 1, 0, 1)
    N = len(spins)
    index = 0
    for i in range(N):
        index += bits[i] * (2 ** i)
    return int(index)
