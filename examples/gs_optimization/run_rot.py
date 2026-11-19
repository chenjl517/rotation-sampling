#!/usr/bin/env python3
"""4x4 PBC J1-J2 Heisenberg ground state — rot-SR with the lookup ansatz.

Usage:
  python run_rot.py --gpu 1
  python run_rot.py --gpu 1 --load ckpts/std_ckpt_heis_4x4_pbc_j1_1.0_j2_0.5_001000.pkl
"""
from __future__ import annotations

import argparse
import os
import time

from common import (  # noqa: E402
    np, ALL_BONDS,
    build_hamiltonian, make_rotation_unitary, rotation_model_state, make_model, make_vstate, make_sr_driver,
    load_ckpt, restore_into, save_ckpt, record, save_history, banner,
)
from config import (  # noqa: E402
    SEED, ROT_SEED, ROT_SWEEPS, INIT_SWEEP, CKPT_EVERY, OUTPUT_DIR, STD_CKPT_FMT, ROT_CKPT_FMT, ROT_HISTORY_PATH,
)
from local_rot_vmc.models.rotated import create_dynamic_u_sz_conserving_rotated_model  # noqa: E402


def run_rot(load_path: str) -> None:
    banner("Rot-SR")

    ckpt = load_ckpt(load_path)
    init_sweep = int(ckpt["config"].get("sweep", INIT_SWEEP))
    base_model = make_model()
    vstate_base = make_vstate(base_model, seed=SEED)
    restore_into(vstate_base, ckpt)
    H_bare = build_hamiltonian()
    print(f"  loaded sweep {init_sweep}: {load_path}")

    U = make_rotation_unitary(ROT_SEED)
    print(f"  rotation U (1 ⊕ O(2) ⊕ 1), O(2) block:\n{np.asarray(U)[1:3, 1:3]}")
    rot_model = create_dynamic_u_sz_conserving_rotated_model(base_model)
    vstate_rot = make_vstate(rot_model, seed=SEED)
    vstate_rot.sampler_state = vstate_base.sampler_state
    n_bonds = len(ALL_BONDS)

    history = [record(init_sweep, vstate_base.expect(H_bare))]
    print(f"  E(start) = {history[0]['energy']:.8f} ± {history[0]['energy_err']:.2e}")
    save_ckpt(ROT_CKPT_FMT, init_sweep, vstate_base, history, init_ckpt=load_path, U=np.asarray(U))

    H_rot_cache: dict[tuple[int, int], object] = {}
    t_start = time.perf_counter()
    for k in range(ROT_SWEEPS):
        sweep = init_sweep + k + 1
        bond, J = ALL_BONDS[k % n_bonds]

        H_rot = H_rot_cache.get(bond)
        if H_rot is None:
            H_rot = H_rot_cache[bond] = build_hamiltonian(rot_bond=bond, U=U)
        vstate_rot.model_state = rotation_model_state(vstate_rot.model_state, bond, U)
        vstate_rot.parameters = {"base": vstate_base.parameters}

        driver = make_sr_driver(H_rot, vstate_rot)
        driver.run(1, show_progress=False)
        vstate_base.parameters = vstate_rot.parameters["base"]

        vstate_base.reset()
        data = record(sweep, vstate_base.expect(H_bare))
        data["rot_bond"] = bond
        history.append(data)
        save_history(history, ROT_HISTORY_PATH, init_sweep=init_sweep, U=np.asarray(U))
        if sweep % CKPT_EVERY == 0 or k == ROT_SWEEPS - 1:
            save_ckpt(ROT_CKPT_FMT, sweep, vstate_base, history, init_ckpt=load_path, U=np.asarray(U))
        if k % 10 == 0 or k == ROT_SWEEPS - 1:
            print(f"  [rot] sweep {sweep:6d}  bond={bond} J={J:g}  "
                  f"E={data['energy']:.8f} ± {data['energy_err']:.2e}  ({time.perf_counter() - t_start:.0f}s)")

    print(f"\n  Rot-SR done in {time.perf_counter() - t_start:.1f}s")
    print(f"  final E = {history[-1]['energy']:.8f} ± {history[-1]['energy_err']:.2e}")
    print(f"  Saved: {ROT_HISTORY_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="4x4 J1-J2 Heisenberg ground state: rot-SR from a run_std.py checkpoint")
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES (applied before JAX init)")
    parser.add_argument("--config", type=str, default=None, metavar="PATH",
                        help="TOML config (default: config.toml next to this script)")
    parser.add_argument("--load", type=str, default=None, metavar="PATH",
                        help=f"run_std.py checkpoint to start from (default: sweep {INIT_SWEEP} of [rotation] init_sweep)")
    args = parser.parse_args()

    load_path = args.load or STD_CKPT_FMT.format(INIT_SWEEP)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"checkpoint not found: {load_path} (run run_std.py first or pass --load PATH)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_rot(load_path)


if __name__ == "__main__":
    main()
