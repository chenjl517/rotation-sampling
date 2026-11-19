#!/usr/bin/env python3
"""Exact dynamics of S+_{site=0}|GS⟩ on the 4x4 PBC J1-J2 XXZ model."""

from __future__ import annotations

import argparse
import os
import sys
import time
import subprocess
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, SCRIPT_DIR)
from config import (  # noqa: E402
    LX, LY, N, PBC, J1, J2, DELTA, EXCI_POS, DT, STD_SWEEPS,
    ED_GS_PATH, ED_DYN_PATH, ED_DIR,
)


def site(x: int, y: int) -> int:
    """Row-major site index consistent with the VMC pipeline."""
    return y * LX + x


def nn_bonds() -> list[tuple[int, int]]:
    """Nearest-neighbor bonds on a rectangular lattice with OBC/PBC."""
    bonds: list[tuple[int, int]] = []
    for y in range(LY):
        for x in range(LX):
            i = site(x, y)
            if PBC or x + 1 < LX:
                j = site((x + 1) % LX, y)
                if i != j:
                    bonds.append((min(i, j), max(i, j)))
            if PBC or y + 1 < LY:
                j = site(x, (y + 1) % LY)
                if i != j:
                    bonds.append((min(i, j), max(i, j)))
    return sorted(set(bonds))


def nnn_bonds() -> list[tuple[int, int]]:
    """Diagonal next-nearest-neighbor bonds on a rectangular lattice with OBC/PBC."""
    bonds: list[tuple[int, int]] = []
    for y in range(LY):
        for x in range(LX):
            i = site(x, y)
            for dx, dy in ((1, 1), (1, -1)):
                if PBC:
                    xn = (x + dx) % LX
                    yn = (y + dy) % LY
                else:
                    xn = x + dx
                    yn = y + dy
                    if not (0 <= xn < LX and 0 <= yn < LY):
                        continue
                j = site(xn, yn)
                if i != j:
                    bonds.append((min(i, j), max(i, j)))
    return sorted(set(bonds))


def build_sz_basis(twice_sz: int) -> np.ndarray:
    """Enumerate 16-bit basis states with fixed 2·Sz_total = twice_sz."""
    if (N - twice_sz) % 2 != 0:
        raise ValueError(f"twice_sz={twice_sz} inconsistent with N={N}")
    n_down = (N - twice_sz) // 2
    if n_down < 0 or n_down > N:
        raise ValueError(f"twice_sz={twice_sz} out of range for N={N}")
    states = np.arange(1 << N, dtype=np.int64)
    popcount = np.array([bin(int(s)).count("1") for s in states], dtype=np.int32)
    return states[popcount == n_down]


def build_hamiltonian_sparse_sector(
    j1: float, j2: float, basis: np.ndarray, delta: float = DELTA,
) -> sparse.csr_matrix:
    """J1-J2 XXZ Hamiltonian restricted to a fixed-Sz basis."""
    dim = len(basis)
    state_to_idx = {int(s): k for k, s in enumerate(basis)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def add_bond(i: int, j: int, J_b: float) -> None:
        mask_i = 1 << i
        mask_j = 1 << j
        for idx, s in enumerate(basis):
            s = int(s)
            si = (s >> i) & 1
            sj = (s >> j) & 1
            szi = 0.5 - si
            szj = 0.5 - sj
            rows.append(idx)
            cols.append(idx)
            vals.append(J_b * delta * szi * szj)
            if si != sj:
                s_new = s ^ mask_i ^ mask_j
                k = state_to_idx.get(s_new)
                if k is not None:
                    rows.append(k)
                    cols.append(idx)
                    vals.append(J_b * 0.5)

    for (i, j) in nn_bonds():
        add_bond(i, j, j1)
    for (i, j) in nnn_bonds():
        add_bond(i, j, j2)

    H = sparse.coo_matrix((vals, (rows, cols)), shape=(dim, dim)).tocsr()
    H = 0.5 * (H + H.T)
    return H


def apply_splus_at_site(psi_sz0: np.ndarray, basis_sz0: np.ndarray,
                        basis_sz1: np.ndarray, site: int) -> np.ndarray:
    """S+_site |ψ⟩ : map Sz=0 amplitudes → Sz=1 amplitudes (unnormalised)."""
    state_to_idx_sz1 = {int(s): k for k, s in enumerate(basis_sz1)}
    psi_sz1 = np.zeros(len(basis_sz1), dtype=np.complex128)
    mask = 1 << site
    for idx0, s in enumerate(basis_sz0):
        s = int(s)
        if (s >> site) & 1 == 1:
            s_new = s & ~mask
            idx1 = state_to_idx_sz1.get(s_new)
            if idx1 is not None:
                psi_sz1[idx1] = psi_sz0[idx0]
    return psi_sz1


def build_sz_i_diagonals(basis: np.ndarray) -> np.ndarray:
    """Diagonal of each Sz_i operator in a given computational basis. Shape (N, dim)."""
    dim = len(basis)
    bits = np.zeros((N, dim), dtype=np.float64)
    for i in range(N):
        bits[i] = np.array([(int(s) >> i) & 1 for s in basis], dtype=np.float64)
    return 0.5 - bits


def format_spin_config(state: int, lx: int = LX, ly: int = LY) -> str:
    """Render a basis state as a row-major lx × ly spin pattern using +/-."""
    rows: list[str] = []
    for y in range(ly):
        row = []
        for x in range(lx):
            site_idx = y * lx + x
            bit = (state >> site_idx) & 1
            row.append("+" if bit == 0 else "-")
        rows.append(" ".join(row))
    return "\n".join(rows)


def save_basis_coeff_data(
    times: np.ndarray,
    psi_traj: np.ndarray,
    psi0: np.ndarray,
    basis_sz1: np.ndarray,
    basis_order: np.ndarray,
) -> tuple[str, str]:
    """Persist ordered coefficient data plus a compact text summary for Codex."""
    coeff_npz = os.path.splitext(ED_DYN_PATH)[0] + "_basis_coeffs.npz"
    coeff_summary = os.path.splitext(ED_DYN_PATH)[0] + "_basis_coeffs_summary.txt"

    coeff_abs = np.abs(psi_traj[:, basis_order]).T.astype(np.float32)
    psi0_abs = np.abs(psi0[basis_order]).astype(np.float64)
    ordered_states = np.asarray(basis_sz1[basis_order], dtype=np.int64)
    np.savez(
        coeff_npz,
        times=np.asarray(times, dtype=np.float64),
        basis_order=np.asarray(basis_order, dtype=np.int64),
        basis_states=ordered_states,
        coeff_abs=coeff_abs,
        psi0_abs=psi0_abs,
    )

    prob_t = np.abs(psi_traj) ** 2
    entropy = -np.sum(np.where(prob_t > 0, prob_t * np.log(prob_t), 0.0), axis=1)
    ipr = 1.0 / np.sum(prob_t**2, axis=1)
    max_prob = np.max(prob_t, axis=1)

    def _rank_line(rank: int, idx: int) -> str:
        state = int(basis_sz1[idx])
        return (
            f"rank {rank:2d}: basis_idx={int(idx):5d} "
            f"|c(0)|={abs(psi0[idx]):.6e} state={state}\n"
            f"{format_spin_config(state)}"
        )

    with open(coeff_summary, "w", encoding="utf-8") as f:
        f.write("Ordered basis-coefficient summary\n")
        f.write(f"times: {times[0]:.6f} ... {times[-1]:.6f}  (n={len(times)})\n")
        f.write(f"basis dimension: {len(basis_sz1)}\n")
        f.write(
            f"t=0: entropy={entropy[0]:.6f}, IPR={ipr[0]:.6f}, "
            f"max_prob={max_prob[0]:.6e}\n"
        )
        f.write(
            f"t_max: entropy={entropy[-1]:.6f}, IPR={ipr[-1]:.6f}, "
            f"max_prob={max_prob[-1]:.6e}\n"
        )
        f.write("\nTop 10 coefficients at t=0:\n")
        for rank, idx in enumerate(basis_order[:10], start=1):
            f.write(_rank_line(rank, int(idx)) + "\n\n")
        f.write("Bottom 10 coefficients at t=0:\n")
        for rank, idx in enumerate(basis_order[-10:][::-1], start=1):
            f.write(_rank_line(rank, int(idx)) + "\n\n")

    return coeff_npz, coeff_summary


def run_codex_analysis(coeff_npz: str, coeff_summary: str) -> None:
    """Invoke the local Codex CLI on the saved coefficient data."""
    prompt = textwrap.dedent(
        f"""
        Analyze the ordered exact-dynamics basis-coefficient data for the 4x4 PBC
        J1-J2 XXZ model. Read the files:
          - {os.path.basename(coeff_npz)}
          - {os.path.basename(coeff_summary)}

        Focus on the distribution of basis coefficients over time. Report:
          1. Whether the initial distribution is localized or spread out.
          2. How quickly the ordered coefficients flatten or broaden.
          3. What the entropy / IPR trend suggests about basis-space support.
          4. Any notable outliers, tails, or symmetry-like patterns in the ordered basis list.

        Keep the answer concise and technical.
        """
    ).strip()

    cmd = ["codex", "exec", "--full-auto", "--ephemeral", "--cd", SCRIPT_DIR, "-"]
    print("  Launching Codex analysis …")
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Codex analysis failed with exit code {proc.returncode}")


def load_or_compute_ed_gs() -> tuple[float, np.ndarray, np.ndarray]:
    if os.path.exists(ED_GS_PATH):
        npz = np.load(ED_GS_PATH)
        E_gs = float(npz["eigvals"][0])
        psi_gs = np.asarray(npz["eigvecs"][:, 0], dtype=np.complex128)
        basis_sz0 = np.asarray(npz["basis"], dtype=np.int64)
        print(f"  Loaded cached ED GS from {ED_GS_PATH}")
        print(f"    E_gs = {E_gs:.10f}, dim = {len(basis_sz0)}")
        return E_gs, psi_gs, basis_sz0

    print(f"  {ED_GS_PATH} missing — computing ED GS now.")
    basis_sz0 = build_sz_basis(twice_sz=0)
    print(f"    Sz=0 dim = {len(basis_sz0)}")
    H0 = build_hamiltonian_sparse_sector(J1, J2, basis_sz0, delta=DELTA)
    t0 = time.time()
    eigvals, eigvecs = eigsh(H0, k=1, which="SA")
    print(f"    eigsh done in {time.time() - t0:.1f}s, E_gs = {eigvals[0]:.10f}")
    psi_gs = eigvecs[:, 0].astype(np.complex128)
    os.makedirs(os.path.dirname(ED_GS_PATH), exist_ok=True)
    np.savez(
        ED_GS_PATH,
        eigvals=eigvals, eigvecs=eigvecs, basis=basis_sz0,
        Lx=LX, Ly=LY, N=N, J1=J1, J2=J2, DELTA=DELTA, pbc=PBC,
        total_sz=0.0, dim=len(basis_sz0),
    )
    print(f"  Cached ED GS → {ED_GS_PATH}")
    return float(eigvals[0]), psi_gs, basis_sz0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", type=str, default=None, metavar="PATH",
                    help="model & ansatz TOML (default: config_model.toml next to this script)")
    ap.add_argument("--config", type=str, default=None, metavar="PATH",
                    help="algorithm TOML providing dt / std_sweeps for the time grid "
                         "(default: tvmc_quench_4x4/config.toml)")
    ap.add_argument(
        "--codex-analyze",
        action="store_true",
        help="save ordered coefficient data and run local Codex analysis afterward",
    )
    args = ap.parse_args()

    print("=" * 70)
    print(f"ED dynamics: 4x4 PBC J1-J2 XXZ  (J1={J1}, J2={J2}, Δ={DELTA})")
    print(f"Quench: S+_{{site={EXCI_POS}}} |GS⟩  →  evolve to t = {STD_SWEEPS * DT:.2f}")
    print("=" * 70)

    os.makedirs(ED_DIR, exist_ok=True)

    E_gs, psi_gs, basis_sz0 = load_or_compute_ed_gs()
    print(f"  |GS| = {np.linalg.norm(psi_gs):.15f}  (should be 1)")

    basis_sz1 = build_sz_basis(twice_sz=2)
    print(f"  Sz=1 dim = {len(basis_sz1)}")
    psi0_raw = apply_splus_at_site(psi_gs, basis_sz0, basis_sz1, site=EXCI_POS)
    n0 = float(np.linalg.norm(psi0_raw))
    print(f"  ||S+_0 |GS⟩|| = {n0:.10f}")
    psi0 = psi0_raw / n0

    print("  Building H in Sz=1 sector …", flush=True)
    t0 = time.time()
    H1 = build_hamiltonian_sparse_sector(J1, J2, basis_sz1, delta=DELTA)
    print(f"    H sparse, nnz = {H1.nnz}, build in {time.time() - t0:.1f}s")

    H1_cplx = H1.astype(np.complex128)
    E0 = float(np.vdot(psi0, H1_cplx @ psi0).real)
    print(f"  ⟨ψ(0)|H|ψ(0)⟩ = {E0:.10f}   (E_gs = {E_gs:.10f}, ΔE = {E0 - E_gs:+.4e})")

    sz_diag = build_sz_i_diagonals(basis_sz1)
    n_times = 1000
    times = DT * np.arange(n_times)

    print(f"  Evolving (expm_multiply) to {n_times} time points (dt={DT}) …", flush=True)
    t0 = time.time()
    psi_traj = expm_multiply(
        -1j * H1_cplx, psi0,
        start=float(times[0]), stop=float(times[-1]),
        num=n_times, endpoint=True,
    )
    print(f"    expm_multiply done in {time.time() - t0:.1f}s")

    prob = np.abs(psi_traj) ** 2
    sz_all = prob @ sz_diag.T

    E_last = float(np.vdot(psi_traj[-1], H1_cplx @ psi_traj[-1]).real)
    norm_last = float(np.linalg.norm(psi_traj[-1]))
    print(f"    ⟨H⟩(t_max) = {E_last:.12f}   (drift {E_last - E0:+.4e})")
    print(f"    ||ψ(t_max)|| = {norm_last:.12f}  (should be 1)")

    np.savez(
        ED_DYN_PATH,
        times=times,
        sz_all=sz_all,
        energy=E0,
        E_gs=E_gs,
        J1=J1, J2=J2, DELTA=DELTA,
        Lx=LX, Ly=LY, N=N,
        exci_pos=EXCI_POS,
    )
    print(f"  Saved → {ED_DYN_PATH}")
    print(f"    times:  shape {times.shape}")
    print(f"    sz_all: shape {sz_all.shape}")
    print(f"    Sz_i(t=0) = {np.array2string(sz_all[0], precision=4)}")

    sz0_png = os.path.splitext(ED_DYN_PATH)[0] + "_sz0.png"
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.plot(times, sz_all[:, EXCI_POS], lw=1.8, color="C3")
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.axhline(1 / 16, color="grey", lw=0.8, ls="--")
    ax.set(
        xlabel="t",
        ylabel=rf"$\langle S^z_{EXCI_POS}\rangle(t)$",
        title=rf"ED quench-site magnetization, $S^z_{EXCI_POS}(t)$",
    )
    ax.grid(alpha=0.3)
    fig.savefig(sz0_png, dpi=160)
    plt.close(fig)
    print(f"  Saved → {sz0_png}")

    basis_order = np.argsort(np.abs(psi0))[::-1]
    print("  Basis ordering for coefficient plot: sorted by |c_n(t=0)| descending")
    print("  Top 10 basis components:")
    for rank, idx in enumerate(basis_order[:10], start=1):
        state = int(basis_sz1[idx])
        print(
            f"    {rank:2d}. basis_idx={int(idx):5d}  "
            f"|c(0)|={abs(psi0[idx]):.6e}  state={state}"
        )
        print(format_spin_config(state))
        print("    " + "-" * 28)

    print("  Bottom 10 basis components:")
    for rank, idx in enumerate(basis_order[-10:][::-1], start=1):
        state = int(basis_sz1[idx])
        print(
            f"    {rank:2d}. basis_idx={int(idx):5d}  "
            f"|c(0)|={abs(psi0[idx]):.6e}  state={state}"
        )
        print(format_spin_config(state))
        print("    " + "-" * 28)

    coeff_png = os.path.splitext(ED_DYN_PATH)[0] + "_basis_coeffs.png"
    coeff_abs = np.abs(psi_traj[:, basis_order]).T
    coeff_log = np.log10(coeff_abs + 1e-16)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    im = ax.imshow(
        coeff_log,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], -0.5, len(basis_sz1) - 0.5],
        cmap="magma",
        vmin=-16.0,
        vmax=float(np.max(coeff_log)),
    )
    tick_pos = np.linspace(0, len(basis_sz1) - 1, 11)
    tick_idx = np.clip(np.round(tick_pos).astype(int), 0, len(basis_sz1) - 1)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([str(int(basis_order[i])) for i in tick_idx])
    ax.set(
        xlabel="t",
        ylabel="basis index (sorted by |c_n(t=0)|)",
        title=r"All basis coefficients in the $S^z=1$ sector: $\log_{10}|c_n(t)|$",
    )
    plt.colorbar(im, ax=ax, label=r"$\log_{10}|c_n(t)|$")
    fig.savefig(coeff_png, dpi=160)
    plt.close(fig)
    print(f"  Saved → {coeff_png}")

    coeff_npz, coeff_summary = save_basis_coeff_data(
        times=times,
        psi_traj=psi_traj,
        psi0=psi0,
        basis_sz1=basis_sz1,
        basis_order=basis_order,
    )
    print(f"  Saved → {coeff_npz}")
    print(f"  Saved → {coeff_summary}")

    if args.codex_analyze:
        run_codex_analysis(coeff_npz, coeff_summary)


if __name__ == "__main__":
    main()
