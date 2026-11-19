#!/usr/bin/env python3
"""Exact diagonalization for the J1-J2 Heisenberg model on a 4x4 square lattice (PBC).

Usage:
    python ed_ground_state.py [--J1 1.0] [--J2 0.5] [--output ed_reference/ed_gs_j2_0.5.npz]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


Lx, Ly = 4, 4
N = Lx * Ly


def site(x: int, y: int) -> int:
    """Row-major site index (consistent with VMC scripts)."""
    return y * Lx + x


def nn_bonds() -> list[tuple[int, int]]:
    """Nearest-neighbour bonds with PBC."""
    bonds = []
    for y in range(Ly):
        for x in range(Lx):
            i = site(x, y)
            j = site((x + 1) % Lx, y)
            bonds.append((min(i, j), max(i, j)))
            j = site(x, (y + 1) % Ly)
            bonds.append((min(i, j), max(i, j)))
    return sorted(set(bonds))


def nnn_bonds() -> list[tuple[int, int]]:
    """Next-nearest-neighbour (diagonal) bonds with PBC."""
    bonds = []
    for y in range(Ly):
        for x in range(Lx):
            i = site(x, y)
            j = site((x + 1) % Lx, (y + 1) % Ly)
            bonds.append((min(i, j), max(i, j)))
            j = site((x + 1) % Lx, (y - 1) % Ly)
            bonds.append((min(i, j), max(i, j)))
    return sorted(set(bonds))


def build_sz0_basis() -> np.ndarray:
    """Enumerate all computational basis states with Sz=0 (equal number of up/down)."""
    half = N // 2
    basis = []
    for s in range(1 << N):
        if bin(s).count("1") == half:
            basis.append(s)
    return np.array(basis, dtype=np.int32)


def build_hamiltonian_sparse(J1: float, J2: float) -> sparse.csr_matrix:
    """Build H = J1 * sum_NN S_i·S_j + J2 * sum_NNN S_i·S_j in the Sz=0 sector as a sparse matrix."""
    basis = build_sz0_basis()
    dim = len(basis)
    print(f"Sz=0 sector dimension: {dim}")

    state_to_idx = {}
    for idx, s in enumerate(basis):
        state_to_idx[int(s)] = idx

    rows = []
    cols = []
    vals = []

    def add_bond(i: int, j: int, J: float):
        mask_i = 1 << i
        mask_j = 1 << j
        for idx, s in enumerate(basis):
            s = int(s)
            si = (s >> i) & 1
            sj = (s >> j) & 1

            szi = 0.5 - si
            szj = 0.5 - sj
            diag = J * szi * szj
            if diag != 0.0:
                rows.append(idx)
                cols.append(idx)
                vals.append(diag)

            if si != sj:
                s_new = s ^ mask_i ^ mask_j
                idx_new = state_to_idx.get(s_new)
                if idx_new is not None:
                    rows.append(idx_new)
                    cols.append(idx)
                    vals.append(J * 0.5)

    all_bonds = [(b, J1) for b in nn_bonds()] + [(b, J2) for b in nnn_bonds()]
    for (i, j), J in all_bonds:
        add_bond(i, j, J)

    H = sparse.coo_matrix((vals, (rows, cols)), shape=(dim, dim))
    H = H.tocsr()
    H = (H + H.T) / 2
    return H


def main():
    parser = argparse.ArgumentParser(description="ED ground state for 4x4 J1-J2 Heisenberg.")
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--J2", type=float, default=0.5)
    parser.add_argument("--n_states", type=int, default=1,
                        help="Number of lowest eigenstates to compute")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz file path (default: ed_reference/ed_gs_j2_{J2}.npz next to this script, "
                             "where run_std.py / run_rot.py look for the reference energy)")
    args = parser.parse_args()

    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(script_dir, "ed_reference", f"ed_gs_j2_{args.J2}.npz")

    print(f"=== ED for 4x4 J1-J2 Heisenberg (PBC) ===")
    print(f"J1 = {args.J1}, J2 = {args.J2}")
    print(f"N = {N} sites, lattice = {Lx}x{Ly}")
    print()

    t0 = time.time()
    H = build_hamiltonian_sparse(args.J1, args.J2)
    t_build = time.time() - t0
    print(f"Hamiltonian built in {t_build:.2f}s  (nnz = {H.nnz})")

    t0 = time.time()
    k = args.n_states
    eigvals, eigvecs = eigsh(H, k=k, which="SA")
    t_diag = time.time() - t0

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    print(f"Diagonalization done in {t_diag:.2f}s")
    print()

    for i in range(k):
        E = eigvals[i]
        print(f"  E[{i}] = {E:.10f}  (E/N = {E / N:.10f})")
    print()

    for i in range(k):
        norm = np.linalg.norm(eigvecs[:, i])
        print(f"  |psi[{i}]| = {norm:.15f}")

    basis = build_sz0_basis()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        eigvals=eigvals,
        eigvecs=eigvecs,
        basis=basis,
        Lx=Lx, Ly=Ly, N=N,
        J1=args.J1, J2=args.J2,
        pbc=True,
        total_sz=0.0,
        dim=len(basis),
    )
    print(f"\nSaved to {output_path}")
    print(f"  eigvals shape: {eigvals.shape}")
    print(f"  eigvecs shape: {eigvecs.shape}")
    print(f"  basis shape:   {basis.shape}")


if __name__ == "__main__":
    main()
