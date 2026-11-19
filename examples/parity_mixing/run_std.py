#!/usr/bin/env python3
"""Parity mixing, plain tVMC reference run (expected to fail: frozen parity).

Usage:
  python run_std.py
  python run_std.py --gpu 4 --config my_params.toml
"""
from __future__ import annotations

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common import (  # noqa: E402
    jax, nk, np,
    HILBERT, build_tfim_hamiltonian, x_sum_operator, make_vstate, thermalize,
    measure_and_record, save_history, save_ckpt, banner, make_driver, evolution_plan,
    build_rbm, initial_rbm_params, load_or_compute_ed,
)

from config import (  # noqa: E402
    SEED, DT, E_GS_SHIFT, INTEGRATOR,
    STD_SWEEPS, OUTPUT_DIR, STD_CKPT_FMT, STD_HISTORY_PATH,
)


def run_evolve_std() -> None:
    banner("Standard tVMC, parity mixing")
    ed = load_or_compute_ed()
    e_gs = float(ed["E_gs"]) if E_GS_SHIFT else 0.0
    print(f"  E_gs = {float(ed['E_gs']):.10f}  (shift {'on' if E_GS_SHIFT else 'off'})")

    model = build_rbm()
    params0 = initial_rbm_params(model, HILBERT)
    H_bare = build_tfim_hamiltonian()
    H_evol = build_tfim_hamiltonian(e_shift=e_gs)
    X_sum = x_sum_operator()

    vstate_measure = make_vstate(model, seed=SEED + 1, params=params0)
    vstate_std = make_vstate(model, seed=SEED + 2, params=params0)
    thermalize(vstate_measure)
    thermalize(vstate_std)

    driver = make_driver(H_evol, vstate_std)
    t_max, max_steps = evolution_plan(STD_SWEEPS)
    print(f"  integrator {INTEGRATOR}: t_max = {t_max:g}, up to {max_steps} steps" + ("" if INTEGRATOR == "rk4" else f", rtol {driver.rtol:g}, dt in [{driver.dt_min:g}, {driver.dt_max:g}], norm {driver.error_norm}") + f", samples {driver.sample_policy}")

    save_ckpt(STD_CKPT_FMT, 0, params0)
    data0 = measure_and_record(vstate_measure, params0, H_bare, X_sum, 0, time=0.0, dt=0.0, rk45_error=0.0, n_rejected=0)
    print(f"  t=0.000  E={data0['energy']:.8f} ± {data0['energy_err']:.1e}  "
          f"P_Z={data0['parity']:+.6f}  ⟨X⟩={data0['x_mean']:+.6f}")
    history = [data0]

    t0 = time.perf_counter()
    sc = 0
    while driver.t < t_max - 1e-12 and sc < max_steps:
        driver.state = vstate_std
        if INTEGRATOR == "rk45":
            driver.dt = min(driver.dt, t_max - driver.t)
        driver.run(1, show_progress=False)
        sc += 1

        save_ckpt(STD_CKPT_FMT, sc, vstate_std.parameters, time=driver.t)
        data = measure_and_record(vstate_measure, vstate_std.parameters, H_bare, X_sum, sc, time=driver.t,
                                  dt=driver.last_dt, rk45_error=driver.last_error, n_rejected=driver.n_rejected)
        history.append(data)
        save_history(history, STD_HISTORY_PATH)
        if (sc - 1) % 10 == 0 or driver.t >= t_max - 1e-12:
            ed_par = float(np.interp(data["time"], ed["times"], ed["parity"]))
            print(f"  [std] step {sc:5d}  t={data['time']:.5f}  dt={driver.last_dt:.2e}  E={data['energy']:.8f}  "
                  f"P_Z={data['parity']:+.6f} (ED {ed_par:+.4f})  ⟨X⟩={data['x_mean']:+.5f}  "
                  f"rej={driver.n_rejected}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    print(f"\n  Standard tVMC done in {time.perf_counter() - t0:.1f}s  ({sc} accepted steps, {driver.n_rejected} rejected)")
    print(f"  Saved: {STD_HISTORY_PATH}  + {sc + 1} checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parity mixing (TFIM): plain tVMC reference run")
    parser.add_argument("--gpu", type=str, default="None",
                        help="CUDA_VISIBLE_DEVICES (applied before JAX init)")
    parser.add_argument("--config", type=str, default=None, metavar="PATH",
                        help="parameter TOML (default: config.toml next to this script)")
    parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_evolve_std()


if __name__ == "__main__":
    main()
