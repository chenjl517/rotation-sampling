#!/usr/bin/env python3
"""Pre-train a bare RBM ansatz to match the ED ground-state amplitudes.

Usage
-----
    python generate_initial_state_gs.py                # default α = 4
    python generate_initial_state_gs.py --rbm-alpha 8
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import netket as nk
import numpy as np
import optax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from config import (  # noqa: E402
    N, J1, J2, DELTA, CKPT_DIR, RBM_ALPHA, RBM_BIAS,
)
from ed_dynamics_heisen import (  # noqa: E402
    load_or_compute_ed_gs,
)


def target_amps_gs() -> tuple[np.ndarray, np.ndarray, float]:
    """Return the ED |GS⟩ amplitudes together with their spin configurations."""
    print("  Loading ED ground state (Sz=0) …", flush=True)
    E_gs, psi_gs, basis_sz0_ed = load_or_compute_ed_gs()
    print(f"    E_gs = {E_gs:.10f},  dim = {len(psi_gs)}")

    bits = (np.asarray(basis_sz0_ed, dtype=np.int64)[:, None] >> np.arange(N)) & 1
    basis_configs = np.where(bits == 1, -1.0, 1.0)
    return np.asarray(psi_gs, dtype=np.complex128), basis_configs, float(E_gs)


def make_rbm(rbm_alpha: int):
    """Bare complex RBM (no Sz projection / ExciWrapper)."""
    return nk.models.RBM(
        param_dtype=jnp.complex128,
        alpha=rbm_alpha,
        use_hidden_bias=RBM_BIAS,
        use_visible_bias=RBM_BIAS,
    )


def fit_rbm(
    target_amps: np.ndarray,
    basis_configs: np.ndarray,
    rbm_alpha: int,
    n_iters: int = 5000,
    lr: float = 0.01,
    seed: int = 0,
    log_every: int = 100,
) -> tuple[dict, list[float], list[float]]:
    """Fit RBM params to maximize fidelity F(θ) = |⟨ψ_RBM(θ)|ψ_target⟩|² / (‖ψ_RBM‖² ‖ψ_target‖²)."""
    model = make_rbm(rbm_alpha)
    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, jnp.asarray(basis_configs[:1]))

    target = jnp.asarray(target_amps, dtype=jnp.complex128)
    target_norm_sq = float(jnp.sum(jnp.abs(target) ** 2).real)
    target_n = target / jnp.sqrt(target_norm_sq)

    basis_jax = jnp.asarray(basis_configs)

    @jax.jit
    def loss_and_metric_complex(params):
        log_psi = model.apply(params, basis_jax)
        log_max = jax.lax.stop_gradient(log_psi.real.max())
        psi = jnp.exp(log_psi - log_max)
        norm_sq = jnp.sum(jnp.abs(psi) ** 2).real
        psi_n = psi / jnp.sqrt(norm_sq)
        overlap = jnp.sum(jnp.conj(psi_n) * target_n)
        fidelity = jnp.abs(overlap) ** 2
        return 1.0 - fidelity, fidelity

    def to_real(params_c):
        return jax.tree.map(lambda x: jnp.stack([x.real, x.imag], axis=-1), params_c)

    def to_complex(params_r):
        return jax.tree.map(lambda x: x[..., 0] + 1j * x[..., 1], params_r)

    params_r = to_real(params)

    @jax.jit
    def loss_and_metric_real(params_r):
        return loss_and_metric_complex(to_complex(params_r))

    if int(os.environ.get("LR_DECAY", "1")):
        schedule = optax.cosine_decay_schedule(
            init_value=lr, decay_steps=n_iters, alpha=0.01,
        )
    else:
        schedule = lr
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params_r)

    @jax.jit
    def step(params_r, opt_state):
        (loss_val, fid), grads = jax.value_and_grad(
            loss_and_metric_real, has_aux=True
        )(params_r)
        updates, opt_state = optimizer.update(grads, opt_state, params_r)
        params_r = optax.apply_updates(params_r, updates)
        return params_r, opt_state, loss_val, fid

    losses, fids = [], []
    t0 = time.time()
    for it in range(n_iters):
        params_r, opt_state, loss_val, fid = step(params_r, opt_state)
        losses.append(float(loss_val))
        fids.append(float(fid))
        if (it == 0) or ((it + 1) % log_every == 0) or (it + 1 == n_iters):
            print(f"  iter {it+1:5d}/{n_iters}  fidelity = {float(fid):.8f}  "
                  f"infid = {float(loss_val):.4e}  ({time.time()-t0:.1f}s)",
                  flush=True)

    params = to_complex(params_r)
    return params, losses, fids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rbm-alpha", type=int, default=RBM_ALPHA,
                    help="RBM hidden density α (n_hidden = α·N)")
    ap.add_argument("--model-config", type=str, default=None, metavar="PATH",
                    help="model & ansatz TOML (default: config_model.toml next to this script)")
    ap.add_argument("--config", type=str, default=None, metavar="PATH",
                    help="algorithm TOML (only needed to pick a different default --rbm-alpha source)")
    ap.add_argument("--n-iters", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 70)
    print(f"RBM_gs amp-fit  α={args.rbm_alpha}  to ED |GS⟩ (Sz=0)  "
          f"iters={args.n_iters}  lr={args.lr}")
    print("=" * 70)

    target_amps, basis_cfg, E_gs = target_amps_gs()
    print(f"  target dim: {len(target_amps)}  "
          f"non-zero: {int(np.sum(np.abs(target_amps) > 1e-15))}  "
          f"||target|| = {np.linalg.norm(target_amps):.6f}")

    params, losses, fids = fit_rbm(
        target_amps, basis_cfg,
        rbm_alpha=args.rbm_alpha,
        n_iters=args.n_iters,
        lr=args.lr, seed=args.seed,
    )
    final_fid = fids[-1]
    print(f"  Final fidelity: {final_fid:.10f}  (infid = {1-final_fid:.4e})")

    tag = f"j2_{J2}_delta_{DELTA}_alpha{args.rbm_alpha}"
    out_pkl = os.path.join(CKPT_DIR, f"rbm_gs_params_{tag}.pkl")
    out_png = os.path.join(CKPT_DIR, f"rbm_gs_fit_{tag}.png")

    os.makedirs(CKPT_DIR, exist_ok=True)
    inner = params["params"] if isinstance(params, dict) and "params" in params else params
    params_np = jax.tree.map(np.asarray, inner)
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "params": params_np,
            "rbm_alpha": args.rbm_alpha,
            "rbm_bias": RBM_BIAS,
            "fidelity": float(final_fid),
            "E_target": E_gs,
            "J1": J1, "J2": J2, "DELTA": DELTA, "N": N,
        }, f)
    print(f"  Saved → {out_pkl}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    its = np.arange(1, len(losses) + 1)
    axes[0].plot(its, fids, lw=1.5, color="C0")
    axes[0].axhline(1.0, color="grey", lw=0.5, ls=":")
    axes[0].set(xlabel="iteration", ylabel="fidelity",
                title=f"RBM_gs amp-fit  α={args.rbm_alpha}  target |GS_ED⟩ (Sz=0)")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[1].semilogy(its, losses, lw=1.5, color="C3")
    axes[1].set(xlabel="iteration", ylabel="1 - fidelity (log)",
                title="infidelity")
    axes[1].grid(alpha=0.3, which="both")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  Saved → {out_png}")


if __name__ == "__main__":
    main()
