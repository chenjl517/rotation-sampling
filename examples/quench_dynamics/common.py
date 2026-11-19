"""Shared setup for the 4x4 PBC J1-J2 XXZ local-quench tVMC demos."""
from __future__ import annotations

import argparse
import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_DIR)

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", default=None)
_pre_args, _ = _pre.parse_known_args()

if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import netket as nk  # noqa: E402
from flax.core import unfreeze  # noqa: E402

from local_rot_vmc.ansatz.exci_wrapper import ExciWrapper  # noqa: E402
from local_rot_vmc.models.rotated import create_sz_projected_model  # noqa: E402
from local_rot_vmc.operators.bond_operator import (  # noqa: E402
    create_1site_rotated_bond_operator,
    create_2site_rotated_bond_operator,
    create_3site_rotated_operator,
)

sys.path.insert(0, SCRIPT_DIR)
from config import (  # noqa: E402
    LX, LY, N, PBC, J1, J2, DELTA, EXCI_POS, SEED, DT, TOTAL_SZ_POST,
    MACHINE_POW, RBM_ALPHA, RBM_BIAS, RBM_GS_PARAMS_PATH, CONFIG_PATH, MODEL_CONFIG_PATH,
)
from ed_dynamics_heisen import nn_bonds, nnn_bonds  # noqa: E402


GRAPH = nk.graph.Grid([LX, LY], pbc=PBC)


def build_excited_model():
    """Build the excited-state Flax *model* ψ = proj_{Sz=TOTAL_SZ_POST}( ExciWrapper(RBM_gs) )."""
    base = nk.models.RBM(
        alpha=RBM_ALPHA, param_dtype=jnp.complex128,
        use_hidden_bias=RBM_BIAS, use_visible_bias=RBM_BIAS,
    )
    excited = ExciWrapper(base=base, exci_pos=EXCI_POS,
                          param_dtype=jnp.complex128)
    return create_sz_projected_model(excited, total_sz=TOTAL_SZ_POST)


def load_rbm_gs_params() -> dict:
    """Load the pre-trained bare RBM_gs *parameters* written by generate_initial_state_gs.py."""
    path = RBM_GS_PARAMS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pre-trained RBM_gs ckpt not found: {path}\n"
            f"Run first: python generate_initial_state_gs.py --rbm-alpha {RBM_ALPHA}"
        )
    with open(path, "rb") as f:
        ck = pickle.load(f)
    ck_bias = ck.get("rbm_bias", True)
    if ck_bias != RBM_BIAS or int(ck.get("rbm_alpha", RBM_ALPHA)) != RBM_ALPHA:
        raise ValueError(
            f"RBM_gs ckpt {path} was fitted with rbm_alpha={ck.get('rbm_alpha')}, "
            f"rbm_bias={ck_bias}, but config_model.toml asks for rbm_alpha={RBM_ALPHA}, "
            f"rbm_bias={RBM_BIAS}. Re-run generate_initial_state_gs.py."
        )
    print(f"  Loaded pre-trained RBM_gs ckpt: {path}")
    print(f"    fidelity = {ck['fidelity']:.10f}   E_gs = {ck['E_target']:.6f}")
    return jax.tree.map(jnp.asarray, ck["params"])


def init_excited_params_from_gs(base_model, gs_params):
    """Build the *parameter values* of the excited model from the RBM_gs params."""
    init_vars = base_model.init(
        jax.random.PRNGKey(SEED),
        jnp.ones((1, N), dtype=jnp.float64),
    )
    params = unfreeze(init_vars["params"])
    params["base"]["base"] = jax.tree.map(jnp.asarray, gs_params)
    return params


def build_j1j2_hamiltonian(
    hilbert,
    *,
    U_local: np.ndarray | None = None,
    active_site: int = EXCI_POS,
    U_bond: np.ndarray | None = None,
    active_bond: tuple[int, int] | None = None,
    e_gs_shift: float = 0.0,
) -> nk.operator.AbstractOperator:
    """J1-J2 XXZ Hamiltonian, optionally rotated by a 1-site or a 2-site unitary."""
    if U_local is not None and U_bond is not None:
        raise ValueError("Pass either U_local (1-site) or U_bond (2-site), not both.")
    if (U_bond is None) != (active_bond is None):
        raise ValueError("U_bond and active_bond must be given together.")
    U_id4 = np.eye(4, dtype=np.complex128)
    U_np = None if U_local is None else np.asarray(U_local)
    U_bond_np = None if U_bond is None else np.asarray(U_bond, dtype=np.complex128)
    bond_set = set(active_bond) if active_bond is not None else set()

    H: nk.operator.AbstractOperator | None = None

    def _add(i: int, j: int, J_b: float) -> None:
        nonlocal H
        delta_b = J_b * DELTA
        if U_np is not None and active_site in (i, j):
            op = create_1site_rotated_bond_operator(
                N=N, i=i, j=j, active_site=active_site,
                model="XXZ", J=J_b, g=0.0, U=U_np,
                hilbert=hilbert, delta=delta_b,
            )
        elif U_bond_np is not None and {i, j} == bond_set:
            op = create_2site_rotated_bond_operator(
                N=N, i=i, j=j, model="XXZ", J=J_b, g=0.0,
                U=U_bond_np, hilbert=hilbert, delta=delta_b,
            )
        elif U_bond_np is not None and len({i, j} & bond_set) == 1:
            op = create_3site_rotated_operator(
                N=N, i=i, j=j, exci_bond=active_bond,
                model="XXZ", J=J_b, g=0.0, U=U_bond_np,
                hilbert=hilbert, delta=delta_b,
            )
        else:
            op = create_2site_rotated_bond_operator(
                N=N, i=i, j=j, model="XXZ", J=J_b, g=0.0,
                U=U_id4, hilbert=hilbert, delta=delta_b,
            )
        H = op if H is None else H + op

    for (i, j) in nn_bonds():
        _add(i, j, J1)
    for (i, j) in nnn_bonds():
        _add(i, j, J2)
    assert H is not None

    if e_gs_shift != 0.0:
        H = H - nk.operator.LocalOperator(
            hilbert, constant=float(e_gs_shift), dtype=np.complex128,
        )
    return H


def measure_and_record(vstate_measure, params, H_bare, sweep_counter: int) -> dict:
    """Measure energy and site-resolved ⟨Sz_i⟩ on all 16 sites."""
    vstate_measure.parameters = params
    vstate_measure.reset()
    E_obs = vstate_measure.expect(H_bare)
    samples = np.asarray(vstate_measure.samples).reshape(-1, N)
    Sz_means = np.mean(samples, axis=0) / 2.0
    return {
        "sweep": sweep_counter,
        "time": sweep_counter * DT,
        "energy": float(E_obs.mean.real),
        "energy_err": float(E_obs.error_of_mean.real),
        "Sz": Sz_means.tolist(),
    }


def save_history(history: list[dict], path: str, **extra) -> None:
    """Write times / sz_all / energies / energy_errs (plus any ``extra`` arrays) to ``path``."""
    times = np.array([h["time"] for h in history])
    sz_all = np.array([h["Sz"] for h in history])
    energies = np.array([h["energy"] for h in history])
    energy_errs = np.array([h["energy_err"] for h in history])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, times=times, sz_all=sz_all,
             energies=energies, energy_errs=energy_errs, **extra)


def save_ckpt(fmt: str, sweep: int, params) -> str:
    params_np = jax.tree.map(np.asarray, params)
    path = fmt.format(sweep)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"params": params_np, "sweep": sweep, "time": sweep * DT}, f)
    return path
