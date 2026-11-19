#!/usr/bin/env python3
"""4x4 PBC J1-J2 XXZ local quench — tVMC with a 1-site local rotation.

Usage:
  python run_rot.py
  python run_rot.py --gpu 0 --config my_params.toml
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
    jax, jnp, np, nk,
    GRAPH, MACHINE_POW,
    build_excited_model, load_rbm_gs_params, init_excited_params_from_gs,
    build_j1j2_hamiltonian, measure_and_record, save_history, save_ckpt,
    CONFIG_PATH, MODEL_CONFIG_PATH,
)
from local_rot_vmc.operators import generate_hadamard_unitary
from local_rot_vmc.models.rotated import create_dynamic_1site_rotated_model
from local_rot_vmc.sr.driver import HolomorphicTDVP, update_model_state_site_and_u
from netket.optimizer.solver import cholesky as nk_cholesky
from local_rot_vmc.sampler import SingleSpinFlipWindowRule
from local_rot_vmc.utils.init_configs import generate_spin_configs

from config import (  # noqa: E402
    N, J1, J2, DELTA, EXCI_POS, SEED, DT, EVOL_SAMPLES, EVOL_CHAINS,
    EVOL_DIAG_SHIFT, KERNEL, ROT_SWEEPS, HELPER_SZ, HELPER_COEFF,
    P_EXCHANGE, OUTPUT_DIR, ROT_CKPT_FMT, ROT_HISTORY_PATH,
)
from ed_dynamics_heisen import load_or_compute_ed_gs  # noqa: E402


def run_evolve_rot() -> None:
    print("=" * 70)
    print(f"Rot-SR evolution  4x4 J1-J2 XXZ  J1={J1} J2={J2} Δ={DELTA}  "
          f"ROT_SWEEPS={ROT_SWEEPS}  dt={DT}")
    print("=" * 70)

    e_gs, _, _ = load_or_compute_ed_gs()
    print(f"  E_gs reference: {e_gs:.10f}")

    hilbert_full = nk.hilbert.Spin(s=0.5, N=N)
    hilbert_sz1 = nk.hilbert.Spin(s=0.5, N=N, total_sz=1)
    chunk_size = max(1, EVOL_CHAINS // jax.device_count())

    gs_params = load_rbm_gs_params()
    base_model = build_excited_model()
    exci_params = init_excited_params_from_gs(base_model, gs_params)
    H_bare = build_j1j2_hamiltonian(hilbert_sz1)

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

    vstate_measure.parameters = exci_params
    vstate_measure.n_discard_per_chain = 20
    vstate_measure.reset()
    vstate_measure.n_discard_per_chain = 1

    _est_samples = jnp.asarray(vstate_measure.samples).reshape(-1, N)
    _est_log_psi = base_model.apply(
        {"params": exci_params, **vstate_measure.model_state}, _est_samples
    )
    amp_scale = float(jnp.exp(jax.scipy.special.logsumexp(jnp.real(_est_log_psi))
                              - jnp.log(_est_log_psi.shape[0])))
    helper_coeff_abs = HELPER_COEFF * amp_scale
    print(f"  ⟨|ψ₀|⟩ ≈ {amp_scale:.4e}  →  helper_coeff_abs = {helper_coeff_abs:.4e}  "
          f"(HELPER_COEFF={HELPER_COEFF} × amp_scale)")

    U_rot = generate_hadamard_unitary(dim=2, dtype=jnp.complex128)
    print(f"  active_site = {EXCI_POS}  (1-site Hadamard rotation, U is 2×2)")

    rot_model = create_dynamic_1site_rotated_model(
        base_model, U_rot,
        helper_model_sz_constraint=HELPER_SZ,
        helper_model_coeff=helper_coeff_abs,
    )
    exchange_rule = nk.sampler.rules.ExchangeRule(graph=GRAPH)
    sflip_rule = SingleSpinFlipWindowRule(sz0=1.0)
    mixed_rule = nk.sampler.rules.MultipleRules(
        rules=(exchange_rule, sflip_rule),
        probabilities=np.array([P_EXCHANGE, 1.0 - P_EXCHANGE]),
    )
    sampler_rot = nk.sampler.MetropolisSampler(
        hilbert=hilbert_full, rule=mixed_rule, n_chains=EVOL_CHAINS,
        sweep_size=2 * N, machine_pow=MACHINE_POW,
    )
    vstate_rot = nk.vqs.MCState(
        sampler=sampler_rot, model=rot_model,
        n_samples=EVOL_SAMPLES, n_discard_per_chain=1,
        seed=SEED, chunk_size=chunk_size,
    )
    vstate_rot.parameters = {"base": exci_params}

    old_sigma = vstate_rot.sampler_state.σ
    init_sigma = jnp.array(
        generate_spin_configs(N=N, total_sz=1, fixed_up_sites=[EXCI_POS],
                              n_configs=old_sigma.shape[0], seed=SEED),
        dtype=old_sigma.dtype,
    ).reshape(old_sigma.shape)
    vstate_rot.sampler_state = vstate_rot.sampler_state.replace(σ=init_sigma)
    vstate_rot.model_state = update_model_state_site_and_u(
        vstate_rot.model_state, EXCI_POS, U_rot,
    )

    vstate_rot.n_discard_per_chain = 20
    _ = vstate_rot.samples
    vstate_rot.n_discard_per_chain = 1

    H_rot = build_j1j2_hamiltonian(
        hilbert_full, U_local=U_rot, active_site=EXCI_POS,
        e_gs_shift=float(e_gs),
    )

    driver_rot = HolomorphicTDVP(
        hamiltonian=H_rot, lr=DT,
        variational_state=vstate_rot, diag_shift=EVOL_DIAG_SHIFT,
        propagation_type="real", kernel=KERNEL,
        linear_solver=nk_cholesky, holomorphic=True,
    )
    driver_rot.set_active_site_and_u(EXCI_POS, U_rot)

    save_ckpt(ROT_CKPT_FMT, 0, exci_params)
    data0 = measure_and_record(vstate_measure, exci_params, H_bare, 0)
    print(f"  t=0.00  E={data0['energy']:.10f} ± {data0['energy_err']:.2e}  "
          f"Sz[{EXCI_POS}]={data0['Sz'][EXCI_POS]:.6f}")
    history = [data0]

    t0 = time.perf_counter()
    for sc in range(ROT_SWEEPS):
        driver_rot.state = vstate_rot
        driver_rot.run(1, show_progress=False)

        curr_params = vstate_rot.parameters["base"]
        save_ckpt(ROT_CKPT_FMT, sc + 1, curr_params)
        data = measure_and_record(vstate_measure, curr_params, H_bare, sc + 1)
        history.append(data)
        print(f"  [rot] sweep {sc+1:3d}  t={data['time']:.4f}  "
              f"E={data['energy']:.10f}  Sz[{EXCI_POS}]={data['Sz'][EXCI_POS]:.6f}")

    print(f"\n  Rot-SR done in {time.perf_counter() - t0:.1f}s")
    save_history(history, ROT_HISTORY_PATH)
    print(f"  Saved: {ROT_HISTORY_PATH}  + {ROT_SWEEPS + 1} checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="4x4 PBC J1-J2 XXZ local quench: tVMC with a 1-site local rotation"
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

    run_evolve_rot()


if __name__ == "__main__":
    main()
