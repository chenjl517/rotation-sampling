#!/usr/bin/env python3
"""4x4 PBC J1-J2 XXZ local quench — p-tVMC with a fixed 2-site local rotation.

Usage:
  python run_rot.py --gpu 0
  python run_rot.py --gpu 1 --config my_params.toml
"""
from __future__ import annotations

import argparse
import os

from ptvmc_common import (  # noqa: E402
    jax, jnp, nk, np,
    make_exchange_vstate, run_ptvmc,
)
from common import (  # noqa: E402
    build_excited_model, load_rbm_gs_params, init_excited_params_from_gs,
    build_j1j2_hamiltonian, CONFIG_PATH, MODEL_CONFIG_PATH,
)
from local_rot_vmc.models.rotated import create_dynamic_u_sz_conserving_rotated_model  # noqa: E402
from local_rot_vmc.operators import generate_hadamard_unitary  # noqa: E402
from local_rot_vmc.sr.driver import update_model_state_bond_and_u  # noqa: E402
from config import (  # noqa: E402
    N, J1, J2, DELTA, EXCI_POS, SEED, DT, ROT_SWEEPS, OUTPUT_DIR,
    ROT_CKPT_FMT, ROT_HISTORY_PATH, ROT_INFIDELITY_PATH,
)
from ed_dynamics_heisen import load_or_compute_ed_gs, nn_bonds  # noqa: E402


def run_ptvmc_rot() -> None:
    print("=" * 70)
    print(f"Rot-p-tVMC  4x4 J1-J2 XXZ  J1={J1} J2={J2} Δ={DELTA}  "
          f"ROT_SWEEPS={ROT_SWEEPS}  dt={DT}")
    print("=" * 70)

    e_gs, _, _ = load_or_compute_ed_gs()
    print(f"  E_gs reference: {e_gs:.10f}")
    hilbert_sz1 = nk.hilbert.Spin(s=0.5, N=N, total_sz=1)

    base_model = build_excited_model()
    init_params = init_excited_params_from_gs(base_model, load_rbm_gs_params())
    H_bare = build_j1j2_hamiltonian(hilbert_sz1)

    rot_bond = min(tuple(b) for b in nn_bonds() if EXCI_POS in b)
    U_rot = generate_hadamard_unitary(dim=4, dtype=jnp.complex128)
    print(f"  active bond = {rot_bond}  (2-site Sz-conserving Hadamard rotation)")

    rot_model = create_dynamic_u_sz_conserving_rotated_model(base_model)
    rot_params = {"base": init_params}

    vstate_measure = make_exchange_vstate(base_model, init_params, seed=SEED + 1)
    vstate_rot = make_exchange_vstate(rot_model, rot_params, seed=SEED + 2)
    vstate_rot.model_state = update_model_state_bond_and_u(
        vstate_rot.model_state, rot_bond, U_rot,
    )

    H_rot = build_j1j2_hamiltonian(
        hilbert_sz1, U_bond=U_rot, active_bond=rot_bond, e_gs_shift=float(e_gs),
    )

    run_ptvmc(
        label="rot", generator=-1j * H_rot,
        vstate=vstate_rot, vstate_measure=vstate_measure, H_bare=H_bare,
        init_params=init_params, n_sweeps=ROT_SWEEPS,
        ckpt_fmt=ROT_CKPT_FMT, history_path=ROT_HISTORY_PATH,
        infidelity_path=ROT_INFIDELITY_PATH,
        params_fn=lambda p: p["base"],
        U_rot=np.asarray(U_rot), rot_bond=np.asarray(rot_bond),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="4x4 PBC J1-J2 XXZ local quench: p-tVMC with a 2-site local rotation"
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

    run_ptvmc_rot()


if __name__ == "__main__":
    main()
