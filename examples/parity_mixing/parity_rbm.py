#!/usr/bin/env python3
"""Exact even-Z-parity initial state in a NetKet RBM (parity-mixing benchmark).

Usage:
  python parity_rbm.py                # build the state, verify it on the full Hilbert space
                                      # (N, rbm_alpha, perturb from config.toml)
"""
from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

from config import (  # noqa: E402
    LX, LY, N, PBC, RBM_ALPHA, RBM_VISIBLE_BIAS, RBM_HIDDEN_BIAS, PERTURB, PERTURB_DEAD, PROJECTOR_UNITS, SEED,
)


def build_rbm(alpha: int = RBM_ALPHA, *, use_visible_bias: bool = RBM_VISIBLE_BIAS,
              use_hidden_bias: bool = RBM_HIDDEN_BIAS) -> nk.models.RBM:
    """Complex RBM, log ψ = Σ a_i x_i + Σ_j log cosh(b_j + Σ_i W_ij x_i)."""
    return nk.models.RBM(alpha=alpha, param_dtype=complex,
                         use_visible_bias=use_visible_bias, use_hidden_bias=use_hidden_bias)


def even_parity_rbm_params(params, n_sites: int, *, perturb: float = PERTURB,
                           perturb_dead: float = PERTURB_DEAD, seed: int = SEED,
                           projector_units: int = PROJECTOR_UNITS):
    """Overwrite ``params`` (NetKet RBM pytree) with an exact even-parity state."""
    W = jnp.zeros_like(params["Dense"]["kernel"])
    b = jnp.zeros_like(params["Dense"]["bias"])
    a = jnp.zeros_like(params["visible_bias"]) if "visible_bias" in params else None
    n_proj = int(projector_units)
    if n_proj not in (1, 2):
        raise ValueError("projector_units must be 1 or 2")
    W = W.at[:, :n_proj].set(-1j * jnp.pi / 4)
    b = b.at[:n_proj].set(1j * jnp.pi * n_sites / 4)
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)

    def noise(k, shape):
        return jax.random.normal(k, shape) + 1j * jax.random.normal(k, shape)

    scale_b = jnp.where(jnp.arange(b.shape[0]) < n_proj, perturb, perturb_dead)
    scale_W = scale_b[None, :]
    W = W + scale_W * noise(keys[0], W.shape)
    b = b + scale_b * noise(keys[1], b.shape)
    out = {"Dense": {"kernel": W, "bias": b}}
    if a is not None:
        out["visible_bias"] = a + perturb_dead * noise(keys[2], a.shape)
    return out


def initial_rbm_params(model: nk.models.RBM, hilbert: nk.hilbert.Spin):
    """Configured initial state: RBM pytree with the exact even-parity construction."""
    init = model.init(jax.random.PRNGKey(SEED), hilbert.random_state(jax.random.PRNGKey(0), 1))["params"]
    return even_parity_rbm_params(init, hilbert.size, perturb=PERTURB, perturb_dead=PERTURB_DEAD,
                                  seed=SEED, projector_units=PROJECTOR_UNITS)


def initial_amplitudes(hilbert: nk.hilbert.Spin) -> np.ndarray:
    """Normalised amplitudes of the configured initial RBM state in hilbert.all_states() order (ED reference)."""
    model = build_rbm()
    logpsi = model.apply({"params": initial_rbm_params(model, hilbert)}, hilbert.all_states())
    psi = np.asarray(jnp.exp(logpsi - jnp.max(logpsi.real)), dtype=np.complex128)
    return psi / np.linalg.norm(psi)


def z_parity(states: np.ndarray) -> np.ndarray:
    """P_Z = Π_i x_i for rows of ±1 configurations."""
    return np.prod(states, axis=1)


def main() -> None:
    hilbert = nk.hilbert.Spin(s=0.5, N=N)
    from lattice import make_graph
    graph = make_graph(LX, LY, PBC)
    model = build_rbm()
    params = model.init(jax.random.PRNGKey(SEED), hilbert.random_state(jax.random.PRNGKey(0), 1))["params"]
    print(f"{LX}x{LY} {'PBC' if PBC else 'OBC'} lattice: N={N}, {graph.n_edges} NN bonds, RBM alpha={RBM_ALPHA} "
          f"({RBM_ALPHA * N} hidden units), dim H = {hilbert.n_states}")

    states = hilbert.all_states()
    parity = z_parity(np.asarray(states))
    proj = (1 + parity) / 2

    cases = (("2 projector units, no noise: equal-weight |ψ_even⟩ of arXiv:2603.18148", 0.0, 0.0, 2),
             ("1 projector unit,  no noise: GHZ_y, signs (−1)^{n_down/2}", 0.0, 0.0, 1),
             (f"config: projector_units={PROJECTOR_UNITS}, perturb={PERTURB}, perturb_dead={PERTURB_DEAD}",
              PERTURB, PERTURB_DEAD, PROJECTOR_UNITS))
    for label, perturb, perturb_dead, n_proj in cases:
        p = even_parity_rbm_params(params, N, perturb=perturb, perturb_dead=perturb_dead, projector_units=n_proj)
        logpsi = model.apply({"params": p}, states)
        psi = np.array(jnp.exp(logpsi - jnp.max(logpsi.real)))
        psi /= np.abs(psi).max()
        prob = np.abs(psi) ** 2
        prob /= prob.sum()
        print(f"\n[{label}]")
        print(f"  max ||ψ(x)|/max|ψ| − proj(x)| = {np.max(np.abs(np.abs(psi) - proj)):.3e}  (0 for equal-magnitude states)")
        print(f"  ⟨P_Z⟩                         = {np.sum(prob * parity):+.12f}")
        print(f"  total probability, odd sector = {prob[parity < 0].sum():.3e}")
        print(f"  finite log ψ on all states    = {bool(np.all(np.isfinite(np.asarray(logpsi.real)[parity > 0])))} (even), "
              f"{bool(np.all(np.isfinite(np.asarray(logpsi.real)[parity < 0])))} (odd)")


if __name__ == "__main__":
    main()
