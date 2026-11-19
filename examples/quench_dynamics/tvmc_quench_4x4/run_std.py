#!/usr/bin/env python3
"""4x4 PBC J1-J2 XXZ local quench — plain tVMC / TDVP reference run.

Usage:
  python run_std.py
  python run_std.py --gpu 1 --config my_params.toml
"""
from __future__ import annotations

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.dirname(SCRIPT_DIR)
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

from common import (  # noqa: E402
    jax, jnp, nk,
    GRAPH, MACHINE_POW,
    build_excited_model, load_rbm_gs_params, init_excited_params_from_gs,
    build_j1j2_hamiltonian, measure_and_record, save_history, save_ckpt,
    CONFIG_PATH, MODEL_CONFIG_PATH,
)
from netket.optimizer.solver import cholesky as nk_cholesky
from local_rot_vmc.sr.driver import HolomorphicTDVP
from local_rot_vmc.utils.init_configs import generate_spin_configs

from config import (  # noqa: E402
    N, J1, J2, DELTA, EXCI_POS, SEED, DT, EVOL_SAMPLES, EVOL_CHAINS,
    EVOL_DIAG_SHIFT, KERNEL, STD_SWEEPS, OUTPUT_DIR, STD_CKPT_FMT, STD_HISTORY_PATH,
)
from ed_dynamics_heisen import load_or_compute_ed_gs  # noqa: E402


def run_evolve_std() -> None:
    print("=" * 70)
    print(f"Standard tVMC  4x4 J1-J2 XXZ  J1={J1} J2={J2} Δ={DELTA}  "
          f"STD_SWEEPS={STD_SWEEPS}  dt={DT}")
    print("=" * 70)

    e_gs, _, _ = load_or_compute_ed_gs()
    print(f"  E_gs reference: {e_gs:.10f}")

    hilbert_sz1 = nk.hilbert.Spin(s=0.5, N=N, total_sz=1)
    chunk_size = max(1, EVOL_CHAINS // jax.device_count())

    base_model = build_excited_model()
    exci_params = init_excited_params_from_gs(base_model, load_rbm_gs_params())
    H_bare = build_j1j2_hamiltonian(hilbert_sz1)
    H_shifted = build_j1j2_hamiltonian(hilbert_sz1, e_gs_shift=float(e_gs))

    vstate_measure = nk.vqs.MCState(
        sampler=nk.sampler.MetropolisExchange(
            hilbert=hilbert_sz1, n_chains=EVOL_CHAINS, graph=GRAPH,
            sweep_size=2 * N, machine_pow=MACHINE_POW),
        model=base_model, n_samples=EVOL_SAMPLES,
        n_discard_per_chain=1, seed=SEED + 1, chunk_size=chunk_size,
    )
    measure_configs = generate_spin_configs(
        N=N, total_sz=1, fixed_up_sites=[EXCI_POS],
        n_configs=EVOL_CHAINS, seed=SEED + 1,
    )
    vstate_measure.sampler_state = vstate_measure.sampler_state.replace(
        σ=jnp.asarray(measure_configs, dtype=vstate_measure.sampler_state.σ.dtype),
    )

    vstate_std = nk.vqs.MCState(
        sampler=nk.sampler.MetropolisExchange(
            hilbert=hilbert_sz1, n_chains=EVOL_CHAINS, graph=GRAPH,
            sweep_size=2 * N, machine_pow=MACHINE_POW),
        model=base_model, n_samples=EVOL_SAMPLES,
        n_discard_per_chain=1, seed=SEED + 2, chunk_size=chunk_size,
    )
    vstate_std.parameters = exci_params
    std_configs = generate_spin_configs(
        N=N, total_sz=1, fixed_up_sites=[EXCI_POS],
        n_configs=EVOL_CHAINS, seed=SEED + 2,
    )
    vstate_std.sampler_state = vstate_std.sampler_state.replace(
        σ=jnp.asarray(std_configs, dtype=vstate_std.sampler_state.σ.dtype),
    )

    vstate_measure.parameters = exci_params
    vstate_measure.n_discard_per_chain = 20
    _ = vstate_measure.samples
    vstate_measure.n_discard_per_chain = 1
    vstate_std.n_discard_per_chain = 20
    _ = vstate_std.samples
    vstate_std.n_discard_per_chain = 1

    driver = HolomorphicTDVP(
        hamiltonian=H_shifted, lr=DT,
        variational_state=vstate_std, diag_shift=EVOL_DIAG_SHIFT,
        propagation_type="real", kernel=KERNEL,
        linear_solver=nk_cholesky, holomorphic=True,
    )

    save_ckpt(STD_CKPT_FMT, 0, exci_params)
    data0 = measure_and_record(vstate_measure, exci_params, H_bare, 0)
    print(f"  t=0.00  E={data0['energy']:.10f} ± {data0['energy_err']:.2e}  "
          f"Sz[{EXCI_POS}]={data0['Sz'][EXCI_POS]:.6f}")
    history = [data0]

    t0 = time.perf_counter()
    for sc in range(STD_SWEEPS):
        driver.state = vstate_std
        driver.run(1, show_progress=False)

        save_ckpt(STD_CKPT_FMT, sc + 1, vstate_std.parameters)
        data = measure_and_record(vstate_measure, vstate_std.parameters, H_bare, sc + 1)
        history.append(data)
        save_history(history, STD_HISTORY_PATH)

        if sc % 10 == 0 or sc == STD_SWEEPS - 1:
            print(f"  [std] sweep {sc+1:6d}  t={data['time']:.4f}  "
                  f"E={data['energy']:.10f}  Sz[{EXCI_POS}]={data['Sz'][EXCI_POS]:.6f}  "
                  f"({time.perf_counter() - t0:.0f}s)")

    print(f"\n  Standard tVMC done in {time.perf_counter() - t0:.1f}s")
    print(f"  Saved: {STD_HISTORY_PATH}  + {STD_SWEEPS + 1} checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="4x4 PBC J1-J2 XXZ local quench: plain tVMC reference run"
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

    run_evolve_std()


if __name__ == "__main__":
    main()
