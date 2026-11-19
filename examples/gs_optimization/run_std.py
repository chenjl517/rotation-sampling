#!/usr/bin/env python3
"""4x4 PBC J1-J2 Heisenberg ground state — standard NetKet VMC-SR with the lookup ansatz.

Usage:
  python run_std.py --gpu 0
  python run_std.py --gpu 0 --config my_params.toml
"""
from __future__ import annotations

import argparse
import os

from common import jax, nk, optax, GRAPH, HILBERT, build_hamiltonian, LookupState  # noqa: E402
from common import PARAM_DTYPE, save_ckpt, history_from_log, record, save_history, banner  # noqa: E402
from config import (  # noqa: E402
    N, TOTAL_SZ, INIT_STDDEV, SEED, N_SAMPLES, N_CHAINS, N_DISCARD, THIN_COEFF, LR, DIAG_SHIFT, GRAD_CLIP,
    STD_SWEEPS, CKPT_EVERY, OUTPUT_DIR, STD_CKPT_FMT, STD_HISTORY_PATH,
)
from netket.optimizer.solver import cholesky  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="4x4 J1-J2 Heisenberg ground state: standard VMC-SR")
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES (applied before JAX init)")
    parser.add_argument("--config", type=str, default=None, metavar="PATH",
                        help="TOML config (default: config.toml next to this script)")
    parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    banner("Standard VMC-SR")

    ha = build_hamiltonian()

    ma = LookupState(N=N, total_sz=TOTAL_SZ, param_dtype=PARAM_DTYPE, init_stddev=INIT_STDDEV)

    sa = nk.sampler.MetropolisExchange(HILBERT, graph=GRAPH, n_chains=N_CHAINS, sweep_size=THIN_COEFF * N)

    vs = nk.vqs.MCState(sa, ma, n_samples=N_SAMPLES, n_discard_per_chain=N_DISCARD, seed=SEED,
                        chunk_size=max(1, N_CHAINS // jax.local_device_count()))
    print(f"  n_parameters = {vs.n_parameters}")

    op = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.sgd(learning_rate=LR))
    gs = nk.driver.VMC_SR(ha, op, variational_state=vs, diag_shift=DIAG_SHIFT, mode="real",
                          linear_solver=cholesky, use_ntk=vs.n_parameters > vs.n_samples)

    log = nk.logging.RuntimeLog()

    def checkpoint(step: int, log_data: dict, driver) -> bool:
        if step % CKPT_EVERY == 0:
            save_ckpt(STD_CKPT_FMT, step, driver.state, history_from_log(log) + [record(step, log_data["Energy"])])
        return True

    gs.run(n_iter=STD_SWEEPS, out=log, callback=checkpoint, show_progress=True)

    vs.reset()
    history = history_from_log(log) + [record(STD_SWEEPS, vs.expect(ha))]
    save_ckpt(STD_CKPT_FMT, STD_SWEEPS, vs, history)
    save_history(history, STD_HISTORY_PATH)
    print(f"\n  final E = {history[-1]['energy']:.8f} ± {history[-1]['energy_err']:.2e}")
    print(f"  Saved: {STD_HISTORY_PATH}")
    print(f"  Checkpoints: {STD_CKPT_FMT.format(0)} ... {STD_CKPT_FMT.format(STD_SWEEPS)}")


if __name__ == "__main__":
    main()
