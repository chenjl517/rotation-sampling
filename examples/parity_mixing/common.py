"""Shared setup for the parity-mixing demo."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from functools import partial

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for _p in (REPO_DIR, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import netket as nk  # noqa: E402
import numpy as np  # noqa: E402

from local_rot_vmc.operators.bond_operator import create_1site_rotated_bond_operator  # noqa: E402
from local_rot_vmc.sampler import DoubleSpinFlipRule  # noqa: E402

from netket.optimizer.solver import cholesky as nk_cholesky, pinv_smooth  # noqa: E402

from config import (  # noqa: E402
    LX, LY, N, PBC, J, H_FIELD, ROT_XY, SEED, DT, EVOL_SAMPLES, EVOL_CHAINS, MACHINE_POW,
    SOLVER, RTOL, RTOL_SMOOTH, CONFIG_PATH, EVOL_DIAG_SHIFT, KERNEL,
    INTEGRATOR, RK45_RTOL, RK45_ATOL, DT_MIN, DT_MAX, T_MAX, MAX_STEPS, SAMPLE_POLICY, RK45_NORM,
)
from ed_dynamics_tfim import build_tfim, load_or_compute_ed  # noqa: E402
from lattice import make_graph  # noqa: E402
from parity_rbm import build_rbm, initial_rbm_params  # noqa: E402,F401

HILBERT = nk.hilbert.Spin(s=0.5, N=N)
GRAPH = make_graph(LX, LY, PBC)
COORDINATION = 2 * GRAPH.n_edges // N


def site(x: int, y: int) -> int:
    """2D lattice coordinate → nk.graph.Grid site index (x * LY + y)."""
    return x * LY + y


ROT_SITE = site(*ROT_XY)


def build_tfim_hamiltonian(hilbert=HILBERT, *, U_local=None, active_site: int | None = None,
                           e_shift: float = 0.0):
    """H = J Σ_<ij> Z_i Z_j + h Σ_i X_i (- e_shift), optionally rotated by a 2x2 U on ``active_site``."""
    if U_local is None:
        H = build_tfim(hilbert, GRAPH, J, H_FIELD)
    else:
        if active_site is None:
            raise ValueError("active_site is required together with U_local")
        U_np = np.asarray(U_local, dtype=np.complex128)
        eye2 = np.eye(2, dtype=np.complex128)
        H = None
        for i, j in GRAPH.edges():
            active, U_ij = (active_site, U_np) if active_site in (i, j) else (i, eye2)
            h_ij = create_1site_rotated_bond_operator(
                N=N, i=i, j=j, active_site=active, model="TFIM",
                J=-J, g=-H_FIELD * 4.0 / COORDINATION, U=U_ij, hilbert=hilbert,
            )
            H = h_ij if H is None else H + h_ij
    if e_shift != 0.0:
        H = H - nk.operator.LocalOperator(hilbert, constant=float(e_shift), dtype=np.complex128)
    return H


def make_solver():
    """Linear solver for the tVMC driver, selected by ``[tvmc] solver`` in the config."""
    if SOLVER == "cholesky":
        return nk_cholesky
    if SOLVER == "pinv_smooth":
        return partial(pinv_smooth, rtol=RTOL, rtol_smooth=RTOL_SMOOTH)
    raise ValueError(f"unknown solver {SOLVER!r} (use 'cholesky' or 'pinv_smooth')")


def x_sum_operator(hilbert=HILBERT):
    """Σ_i X_i (site-averaged ⟨X_i⟩ = ⟨X_sum⟩ / N)."""
    return sum(nk.operator.spin.sigmax(hilbert, i) for i in range(hilbert.size))


def make_driver(hamiltonian, vstate):
    """RK4 / RK45 TDVP driver with the [tvmc] settings (real-time, holomorphic, configured kernel and solver)."""
    from local_rot_vmc.sr.driver import HolomorphicTDVP
    return HolomorphicTDVP(
        hamiltonian=hamiltonian, lr=DT, variational_state=vstate, diag_shift=EVOL_DIAG_SHIFT,
        propagation_type="real", kernel=KERNEL, linear_solver=make_solver(), holomorphic=True,
        integrator=INTEGRATOR, rtol=RK45_RTOL, atol=RK45_ATOL, dt_min=DT_MIN, dt_max=DT_MAX,
        sample_policy=SAMPLE_POLICY, error_norm=RK45_NORM,
    )


def evolution_plan(sweeps: int) -> tuple[float, int]:
    """(t_max, max accepted steps) of a run: rk4 → sweeps * dt; rk45 → [tvmc] t_max (fallback sweeps * dt)."""
    if INTEGRATOR == "rk4":
        return sweeps * DT, sweeps
    return (T_MAX if T_MAX > 0 else sweeps * DT), MAX_STEPS


def make_vstate(model, *, seed: int, params=None) -> nk.vqs.MCState:
    """MCState with the paper's Metropolis sampler: 50% single-spin flips + 50% double flips."""
    chunk_size = max(1, EVOL_CHAINS // jax.device_count())
    rule = nk.sampler.rules.MultipleRules(
        rules=(nk.sampler.rules.LocalRule(), DoubleSpinFlipRule()), probabilities=np.array([0.5, 0.5]),
    )
    sampler = nk.sampler.MetropolisSampler(
        hilbert=HILBERT, rule=rule, n_chains=EVOL_CHAINS, sweep_size=N, machine_pow=MACHINE_POW,
    )
    vstate = nk.vqs.MCState(
        sampler=sampler, model=model, n_samples=EVOL_SAMPLES,
        n_discard_per_chain=1, seed=seed, chunk_size=chunk_size,
    )
    if params is not None:
        vstate.parameters = params
    return vstate


def thermalize(vstate: nk.vqs.MCState, n_discard: int = 20) -> None:
    vstate.n_discard_per_chain = n_discard
    vstate.reset()
    _ = vstate.samples
    vstate.n_discard_per_chain = 1


def measure_and_record(vstate_measure, params, H_bare, X_sum, step: int, time: float | None = None,
                       **extra) -> dict:
    """Energy, Z-parity ⟨Π_i Z_i⟩ (diagonal → plain sample average) and site-averaged ⟨X_i⟩."""
    vstate_measure.parameters = params
    vstate_measure.reset()
    E_obs = vstate_measure.expect(H_bare)
    X_obs = vstate_measure.expect(X_sum)
    samples = np.asarray(vstate_measure.samples).reshape(-1, N)
    parity = np.prod(samples, axis=1)
    return {
        "step": step,
        "time": step * DT if time is None else float(time),
        **extra,
        "energy": float(E_obs.mean.real),
        "energy_err": float(E_obs.error_of_mean.real),
        "parity": float(parity.mean()),
        "parity_err": float(parity.std(ddof=1) / np.sqrt(parity.size)),
        "x_mean": float(X_obs.mean.real) / N,
    }


def save_history(history: list[dict], path: str, **extra) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        times=np.array([h["time"] for h in history]),
        parity=np.array([h["parity"] for h in history]),
        parity_err=np.array([h["parity_err"] for h in history]),
        x_mean=np.array([h["x_mean"] for h in history]),
        energies=np.array([h["energy"] for h in history]),
        energy_errs=np.array([h["energy_err"] for h in history]),
        **({"dt": np.array([h["dt"] for h in history])} if all("dt" in h for h in history) else {}),
        **extra,
    )


def save_ckpt(fmt: str, step: int, params, time: float | None = None) -> str:
    params_np = jax.tree.map(np.asarray, params)
    path = fmt.format(step)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"params": params_np, "step": step, "time": step * DT if time is None else float(time)}, f)
    return path


def banner(title: str) -> None:
    print("=" * 70)
    print(f"{title}  {LX}x{LY} {'PBC' if PBC else 'OBC'} TFIM  J={J} h={H_FIELD}  dt={DT}")
    print("=" * 70)
    print(f"  Backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(f"  Config:  {CONFIG_PATH}")
