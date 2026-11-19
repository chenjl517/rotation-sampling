#!/usr/bin/env python3
"""4x4 PBC J1-J2 XXZ local quench — plain p-tVMC reference run.

Usage:
  python run_std.py --gpu 0
  python run_std.py --gpu 1 --config my_params.toml
"""
from __future__ import annotations

import argparse
import os

from ptvmc_common import (  # noqa: E402
    jax, nk, np,
    make_exchange_vstate, run_ptvmc,
)
from common import (  # noqa: E402
    build_excited_model, load_rbm_gs_params, init_excited_params_from_gs,
    build_j1j2_hamiltonian, CONFIG_PATH, MODEL_CONFIG_PATH,
)
from config import (  # noqa: E402
    N, J1, J2, DELTA, SEED, DT, STD_SWEEPS, OUTPUT_DIR,
    STD_CKPT_FMT, STD_HISTORY_PATH, STD_INFIDELITY_PATH,
)
from ed_dynamics_heisen import load_or_compute_ed_gs  # noqa: E402


def run_ptvmc_std() -> None:
    print("=" * 70)
    print(f"p-tVMC  4x4 J1-J2 XXZ  J1={J1} J2={J2} Δ={DELTA}  "
          f"STD_SWEEPS={STD_SWEEPS}  dt={DT}")
    print("=" * 70)

    e_gs, _, _ = load_or_compute_ed_gs()
    print(f"  E_gs reference: {e_gs:.10f}")
    hilbert_sz1 = nk.hilbert.Spin(s=0.5, N=N, total_sz=1)

    base_model = build_excited_model()
    init_params = init_excited_params_from_gs(base_model, load_rbm_gs_params())
    H_bare = build_j1j2_hamiltonian(hilbert_sz1)
    H_shifted = build_j1j2_hamiltonian(hilbert_sz1, e_gs_shift=float(e_gs))

    vstate_measure = make_exchange_vstate(base_model, init_params, seed=SEED + 1)
    vstate = make_exchange_vstate(base_model, init_params, seed=SEED + 2)

    run_ptvmc(
        label="std", generator=-1j * H_shifted,
        vstate=vstate, vstate_measure=vstate_measure, H_bare=H_bare,
        init_params=init_params, n_sweeps=STD_SWEEPS,
        ckpt_fmt=STD_CKPT_FMT, history_path=STD_HISTORY_PATH,
        infidelity_path=STD_INFIDELITY_PATH,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="4x4 PBC J1-J2 XXZ local quench: plain p-tVMC reference run"
    )
    parser.add_argument("--gpu", type=str, default="None",
                        help="CUDA_VISIBLE_DEVICES (applied before JAX init)")
    parser.add_argument("--config", type=str, default=None, metavar="PATH",
                        help="algorithm TOML (default: config.toml next to this script)")
    parser.add_argument("--model-config", type=str, default=None, metavar="PATH",
                        help="model & ansatz TOML (default: ../config_model.toml)")
    parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Model config:     {MODEL_CONFIG_PATH}")
    print(f"Algorithm config: {CONFIG_PATH}\n")

    run_ptvmc_std()


if __name__ == "__main__":
    main()
