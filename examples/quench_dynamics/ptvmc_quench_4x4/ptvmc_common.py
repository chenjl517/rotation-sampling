"""Shared p-tVMC plumbing for run_rot.py / run_std.py."""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.dirname(SCRIPT_DIR)
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

from common import (  # noqa: E402
    jax, jnp, np, nk,
    GRAPH, MACHINE_POW,
    measure_and_record, save_history, save_ckpt,
)
from local_rot_vmc.utils.init_configs import generate_spin_configs  # noqa: E402
from netket.optimizer.solver import cholesky as nk_cholesky  # noqa: E402

from config import (  # noqa: E402
    N, EXCI_POS, SEED, DT, EVOL_SAMPLES, EVOL_CHAINS,
    PTVMC_SOLVER, PTVMC_N_ITER, PTVMC_LR, PTVMC_DIAG_SHIFT,
    PTVMC_AUTOTUNE, PTVMC_AUTOTUNE_TARGET, PTVMC_DIAG_SHIFT_MIN, PTVMC_DIAG_SHIFT_MAX,
    PTVMC_SAMPLE_UPHI,
)

try:
    import optax  # noqa: E402
    import ptvmc  # noqa: E402
    import advanced_drivers as advd  # noqa: E402
    from advanced_drivers._src.callbacks.autodiagshift import PI_controller_diagshift  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The p-tVMC demos need the external ptvmc-systematic-study package "
        f"(missing: {exc.name}). Install it with:\n"
        "  pip install git+https://github.com/NeuralQXLab/ptvmc-systematic-study"
    ) from exc


class _NoProgressBar:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, step, log_data, driver) -> bool:
        return True


ptvmc._src.compression.infidelity.ProgressBarCallback = _NoProgressBar


def make_exchange_vstate(model, params, *, seed: int):
    """MCState in the Sz=+1 sector with the NN Metropolis-exchange sampler."""
    hilbert_sz1 = nk.hilbert.Spin(s=0.5, N=N, total_sz=1)
    chunk_size = max(1, EVOL_CHAINS // jax.device_count())
    vstate = nk.vqs.MCState(
        sampler=nk.sampler.MetropolisExchange(
            hilbert=hilbert_sz1, n_chains=EVOL_CHAINS, graph=GRAPH,
            sweep_size=2 * N, machine_pow=MACHINE_POW),
        model=model, n_samples=EVOL_SAMPLES,
        n_discard_per_chain=1, seed=seed, chunk_size=chunk_size,
    )
    vstate.parameters = params
    configs = generate_spin_configs(
        N=N, total_sz=1, fixed_up_sites=[EXCI_POS],
        n_configs=EVOL_CHAINS, seed=seed,
    )
    vstate.sampler_state = vstate.sampler_state.replace(
        σ=jnp.asarray(configs, dtype=vstate.sampler_state.σ.dtype),
    )
    vstate.n_discard_per_chain = 20
    _ = vstate.samples
    vstate.n_discard_per_chain = 1
    return vstate


def make_solver():
    """Product-expansion scheme named by PTVMC_SOLVER (e.g. ``ptvmc.solver.LPE3()``)."""
    solver_factory = getattr(ptvmc.solver, PTVMC_SOLVER, None)
    if solver_factory is None or PTVMC_SOLVER.startswith("S"):
        raise SystemExit(
            f"Unsupported p-tVMC solver {PTVMC_SOLVER!r}; use LPE1/2/3 or PPE2/4/6."
        )
    return solver_factory()


def make_ptvmc_driver(generator, vstate, solver, inner_callbacks: list[Callable] = ()):
    """PTVMCDriver for ``generator = -i (H - E_gs)`` with the given solver."""
    callbacks: list[Any] = []
    if PTVMC_AUTOTUNE:
        callbacks.append(PI_controller_diagshift(
            target=PTVMC_AUTOTUNE_TARGET, safety_fac=1.0,
            clip_min=0.5, clip_max=2.0,
            diag_shift_min=PTVMC_DIAG_SHIFT_MIN, diag_shift_max=PTVMC_DIAG_SHIFT_MAX,
            order=1, beta_1=0.9, beta_2=0.1,
        ))
    callbacks.extend(inner_callbacks)

    compression = ptvmc.compression.InfidelityCompression(
        driver_class=advd.driver.InfidelityOptimizerNG,
        build_parameters={
            "diag_shift": PTVMC_DIAG_SHIFT,
            "optimizer": optax.inject_hyperparams(optax.sgd)(learning_rate=PTVMC_LR),
            "linear_solver_fn": nk_cholesky,
            "collect_quadratic_model": True,
            "cv_coeff": -0.5,
            "estimator": "cmc",
            "sample_Uphi": PTVMC_SAMPLE_UPHI,
        },
        run_parameters={"n_iter": PTVMC_N_ITER, "callback": callbacks},
    )
    return ptvmc.PTVMCDriver(
        generator, 0.0,
        solver=solver,
        integration_params=ptvmc.IntegrationParameters(dt=float(DT)),
        compression_algorithm=compression,
        variational_state=vstate,
    )


def _last_infidelity(log_data: Any) -> float:
    found: list[float] = []

    def _walk(node: Any) -> None:
        if not hasattr(node, "items"):
            return
        for key, child in node.items():
            if str(key).lower().split("/")[-1] == "infidelity":
                mean = getattr(child, "mean", child)
                found.append(float(np.real(np.ravel(np.asarray(mean))[-1])))
            _walk(child)

    _walk(log_data)
    return found[-1] if found else float("nan")


class InfidelityRecorder:
    """Inner-loop callback: infidelity after every optimisation iteration."""

    def __init__(self, path: str, *, n_iter: int, stages: int, flush_every: int = 50):
        self.path = path
        self.n_iter = int(n_iter)
        self.stages = int(stages)
        self.flush_every = int(flush_every)
        self.infidelity: list[float] = []

    def __call__(self, step: int, log_data: dict, driver: Any) -> bool:
        self.infidelity.append(_last_infidelity(log_data))
        if len(self.infidelity) % self.flush_every == 0:
            self.flush()
        return True

    def flush(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        np.savez(
            self.path,
            step=np.arange(len(self.infidelity), dtype=np.int64),
            infidelity=np.asarray(self.infidelity, dtype=np.float64),
            n_iter=self.n_iter, stages=self.stages,
        )

    def last(self) -> float:
        return self.infidelity[-1] if self.infidelity else float("nan")


class HistoryRecorder:
    """Outer-loop callback: checkpoint + energy / ⟨Sz_i⟩ after each time step."""

    def __init__(self, *, vstate_measure, H_bare, ckpt_fmt: str, history_path: str,
                 history: list[dict], infidelity: InfidelityRecorder,
                 params_fn: Callable = lambda p: p, label: str = "ptvmc",
                 print_every: int = 10, **history_extra):
        self.vstate_measure = vstate_measure
        self.H_bare = H_bare
        self.ckpt_fmt = ckpt_fmt
        self.history_path = history_path
        self.history = history
        self.infidelity = infidelity
        self.params_fn = params_fn
        self.label = label
        self.print_every = print_every
        self.history_extra = history_extra
        self.t0 = time.perf_counter()

    def __call__(self, step: int, log_data: dict, driver: Any) -> bool:
        sweep = int(round(float(driver.t) / DT))
        params = self.params_fn(driver.state.parameters)
        save_ckpt(self.ckpt_fmt, sweep, params)
        data = measure_and_record(self.vstate_measure, params, self.H_bare, sweep)
        self.history.append(data)
        save_history(self.history, self.history_path, **self.history_extra)
        if sweep == 1 or sweep % self.print_every == 0:
            print(f"  [{self.label}] sweep {sweep:6d}  t={data['time']:.4f}  "
                  f"E={data['energy']:.10f}  Sz[{EXCI_POS}]={data['Sz'][EXCI_POS]:.6f}  "
                  f"Infid={self.infidelity.last():.3e}  "
                  f"({time.perf_counter() - self.t0:.0f}s)", flush=True)
        return True


def run_ptvmc(*, label: str, generator, vstate, vstate_measure, H_bare, init_params,
              n_sweeps: int, ckpt_fmt: str, history_path: str, infidelity_path: str,
              params_fn: Callable = lambda p: p, **history_extra) -> None:
    """Evolve ``vstate`` under ``generator`` for ``n_sweeps`` steps of DT and record."""
    solver = make_solver()
    stages = max(1, int(getattr(solver, "stages", 1)))
    infidelity = InfidelityRecorder(infidelity_path, n_iter=PTVMC_N_ITER, stages=stages)
    driver = make_ptvmc_driver(generator, vstate, solver, [infidelity])

    save_ckpt(ckpt_fmt, 0, init_params)
    data0 = measure_and_record(vstate_measure, init_params, H_bare, 0)
    history = [data0]
    save_history(history, history_path, **history_extra)
    print(f"  t=0.00  E={data0['energy']:.10f} ± {data0['energy_err']:.2e}  "
          f"Sz[{EXCI_POS}]={data0['Sz'][EXCI_POS]:.6f}")

    recorder = HistoryRecorder(
        vstate_measure=vstate_measure, H_bare=H_bare, ckpt_fmt=ckpt_fmt,
        history_path=history_path, history=history, infidelity=infidelity,
        params_fn=params_fn, label=label, **history_extra,
    )
    t0 = time.perf_counter()
    driver.run(
        T=float(n_sweeps * DT), out=nk.logging.RuntimeLog(),
        show_progress=False, callback=[recorder],
    )
    infidelity.flush()
    print(f"\n  p-tVMC ({label}) done in {time.perf_counter() - t0:.1f}s")
    print(f"  Saved: {history_path}")
    print(f"  Saved: {infidelity_path}  + {n_sweeps + 1} checkpoints")
