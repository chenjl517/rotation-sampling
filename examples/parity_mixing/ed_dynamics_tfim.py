#!/usr/bin/env python3
"""Exact reference dynamics for the parity-mixing benchmark (TFIM on the configured lattice).

Usage:
  python ed_dynamics_tfim.py                       # parameters from config.toml
  python ed_dynamics_tfim.py --config other.toml
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
from scipy.sparse.linalg import eigsh, expm_multiply

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from config import LX, LY, N, PBC, J, H_FIELD, ED_DT, ED_T_MAX, ED_DIR, ED_DYN_PATH, _INIT_TAG  # noqa: E402
from parity_rbm import initial_amplitudes  # noqa: E402
from lattice import make_graph  # noqa: E402


def build_tfim(hilbert: nk.hilbert.Spin, graph: nk.graph.Graph, J: float, h: float) -> nk.operator.LocalOperator:
    """H = J Σ_<ij> Z_i Z_j + h Σ_i X_i with Pauli matrices."""
    sz, sx = nk.operator.spin.sigmaz, nk.operator.spin.sigmax
    H = sum(J * (sz(hilbert, i) @ sz(hilbert, j)) for i, j in graph.edges())
    H = H + sum(h * sx(hilbert, i) for i in range(hilbert.size))
    return H


def even_parity_state(hilbert: nk.hilbert.Spin) -> tuple[np.ndarray, np.ndarray]:
    """Normalised equal-weight |ψ_even⟩ in the hilbert.all_states() ordering, and P_Z of every basis state."""
    parity = np.prod(np.asarray(hilbert.all_states()), axis=1)
    psi0 = ((1 + parity) / 2).astype(np.complex128)
    return psi0 / np.linalg.norm(psi0), parity


def compute_ed(out_path: str = ED_DYN_PATH, *, plot: bool = True) -> dict:
    hilbert = nk.hilbert.Spin(s=0.5, N=N)
    graph = make_graph(LX, LY, PBC)
    print(f"  ED: {LX}x{LY} {'PBC' if PBC else 'OBC'} TFIM  J={J} h={H_FIELD}  "
          f"N={N}  bonds={graph.n_edges}  dim={hilbert.n_states}")

    t0 = time.time()
    H = build_tfim(hilbert, graph, J, H_FIELD).to_sparse().tocsc()
    X_sum = sum(nk.operator.spin.sigmax(hilbert, i) for i in range(N)).to_sparse().tocsc()
    psi_eq, parity = even_parity_state(hilbert)
    psi0 = initial_amplitudes(hilbert)
    odd_weight = float(np.sum(np.abs(psi0[parity < 0]) ** 2))
    print(f"  initial state = rbm ({_INIT_TAG}):  |⟨ψ_eq|ψ_0⟩| = {abs(np.vdot(psi_eq, psi0)):.6f},  "
          f"odd-sector weight = {odd_weight:.2e}")
    zz_sum = np.asarray(sum(nk.operator.spin.sigmaz(hilbert, i) @ nk.operator.spin.sigmaz(hilbert, j)
                            for i, j in graph.edges()).to_sparse().diagonal()).real
    e_gs = float(eigsh(H, k=1, which="SA")[0][0])
    print(f"  operators built in {time.time() - t0:.1f}s;  E_gs = {e_gs:.10f};  "
          f"⟨P_Z⟩(0) = {np.vdot(psi0, parity * psi0).real:.6f}")

    n_steps = int(round(ED_T_MAX / ED_DT))
    times = ED_DT * np.arange(n_steps + 1)
    t0 = time.time()
    traj = expm_multiply(-1j * H, psi0, start=0.0, stop=ED_T_MAX, num=n_steps + 1, endpoint=True)
    print(f"  expm_multiply: {n_steps + 1} time points in {time.time() - t0:.1f}s")

    norms = np.linalg.norm(traj, axis=1)
    traj = traj / norms[:, None]
    par_t = np.einsum("ti,i,ti->t", traj.conj(), parity, traj).real
    x_mean = np.einsum("ti,ti->t", traj.conj(), (X_sum @ traj.T).T).real / N
    zz_mean = np.einsum("ti,i,ti->t", traj.conj(), zz_sum, traj).real / graph.n_edges
    energy = np.einsum("ti,ti->t", traj.conj(), (H @ traj.T).T).real
    print(f"  max |norm−1| = {np.abs(norms - 1).max():.2e},  energy drift = {np.ptp(energy):.2e}")
    print(f"  P_Z(t): 1 → min {par_t.min():+.4f} at t={times[par_t.argmin()]:.2f};  "
          f"P_Z(t_max) = {par_t[-1]:+.4f}")

    data = dict(times=times, parity=par_t, x_mean=x_mean, zz_mean=zz_mean,
                energy=energy[0], E_gs=e_gs, J=J, h=H_FIELD, N=N)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **data)
    print(f"  Saved: {out_path}")
    if plot:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(times, par_t, "k-", lw=1, label=r"$\langle P_Z\rangle$")
        ax.plot(times, x_mean, "C0--", lw=1, label=r"$\langle X_i\rangle$")
        ax.set(xlabel="t", title=f"{LX}x{LY} PBC TFIM, J={J}, h={H_FIELD}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path.replace(".npz", ".png"), dpi=120)
    return data


def load_or_compute_ed() -> dict:
    """Reference dynamics + E_gs for the configured (J, h); computed on first use."""
    if os.path.exists(ED_DYN_PATH):
        d = np.load(ED_DYN_PATH)
        if "E_gs" in d.files:
            return {k: d[k] for k in d.files}
    print(f"  ED reference {ED_DYN_PATH} missing → computing")
    return compute_ed()


def main() -> None:
    parser = argparse.ArgumentParser(description="ED reference dynamics: TFIM from |ψ_even⟩")
    parser.add_argument("--config", type=str, default=None, metavar="PATH",
                        help="TOML parameters (default: config.toml next to this script)")
    parser.parse_args()
    os.makedirs(ED_DIR, exist_ok=True)
    compute_ed()


if __name__ == "__main__":
    main()
