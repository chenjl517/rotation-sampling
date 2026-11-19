"""Shared setup for the lookup-ansatz ground-state demo."""
from __future__ import annotations

import argparse
import os
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
import jax.numpy as jnp  # noqa: E402
import netket as nk  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from local_rot_vmc import extract_bonds_from_graph, get_2d_nnn_bonds, load_checkpoint, save_checkpoint  # noqa: E402
from local_rot_vmc.ansatz.lookup_state import LookupState  # noqa: E402
from local_rot_vmc.operators import create_2site_rotated_bond_operator, create_3site_rotated_operator  # noqa: E402
from local_rot_vmc.operators.unitary import random_unitary_block  # noqa: E402
from local_rot_vmc.sr.driver import _update_u_info_recursive, update_model_state_bond  # noqa: E402

from config import (  # noqa: E402
    LX, N, PBC, J1, J2, TOTAL_SZ, INIT_STDDEV, SEED, N_SAMPLES, N_CHAINS, N_DISCARD, THIN_COEFF,
    LR, DIAG_SHIFT, SOLVER, GRAD_CLIP, CONFIG_PATH, ED_GS_PATH,
)

PARAM_DTYPE = jnp.float32
NP_PARAM_DTYPE = np.float32

HILBERT = nk.hilbert.Spin(s=0.5, N=N, total_sz=TOTAL_SZ)
GRAPH = nk.graph.Square(length=LX, pbc=PBC)
NN_BONDS = extract_bonds_from_graph(GRAPH)
NNN_BONDS = get_2d_nnn_bonds(LX, pbc=PBC) if J2 != 0.0 else []
ALL_BONDS: list[tuple[tuple[int, int], float]] = [(b, J1) for b in NN_BONDS] + [(b, J2) for b in NNN_BONDS]


def build_hamiltonian(*, rot_bond: tuple[int, int] | None = None, U=None):
    """H = J1 Σ_NN S_i·S_j + J2 Σ_NNN S_i·S_j, optionally rotated by a 4x4 U on ``rot_bond``."""
    eye4 = np.identity(4, dtype=NP_PARAM_DTYPE)
    U_np = None if U is None else np.asarray(U, dtype=NP_PARAM_DTYPE)
    rot_set = set(rot_bond) if rot_bond is not None else set()
    H = None
    for (i, j), J in ALL_BONDS:
        overlap = rot_set & {i, j}
        if rot_bond is not None and {i, j} == rot_set:
            op = create_2site_rotated_bond_operator(
                N=N, i=rot_bond[0], j=rot_bond[1], model="Heisenberg", J=J, g=0.0, U=U_np, hilbert=HILBERT)
        elif len(overlap) == 1:
            op = create_3site_rotated_operator(
                N=N, i=i, j=j, exci_bond=rot_bond, model="Heisenberg", J=J, g=0.0, U=U_np, hilbert=HILBERT)
        else:
            op = create_2site_rotated_bond_operator(
                N=N, i=i, j=j, model="Heisenberg", J=J, g=0.0, U=eye4, hilbert=HILBERT)
        H = op if H is None else H + op
    return H


def make_rotation_unitary(seed: int) -> jnp.ndarray:
    """Random real Sz-conserving 4x4 rotation 1 ⊕ O(2) ⊕ 1 in the basis (↑↑, ↑↓, ↓↑, ↓↓)."""
    O2 = np.asarray(random_unitary_block(jax.random.PRNGKey(seed), 2, dtype=PARAM_DTYPE))
    U = np.identity(4, dtype=NP_PARAM_DTYPE)
    U[1:3, 1:3] = O2
    if not np.allclose(U.T @ U, np.identity(4), atol=1e-5):
        raise RuntimeError("rotation matrix is not orthogonal")
    return jnp.asarray(U, dtype=PARAM_DTYPE)


def rotation_model_state(model_state, bond: tuple[int, int], U) -> dict:
    """model_state of the dynamic rotated model with the active ``bond`` and the U† blocks of ``U``."""
    U_dag = jnp.asarray(U, dtype=PARAM_DTYPE).T
    ms = update_model_state_bond(model_state, bond[0], bond[1])
    u_info = {
        "phi1": U_dag[0, 0], "phi2": U_dag[3, 3],
        "block_00": U_dag[1, 1], "block_01": U_dag[1, 2],
        "block_10": U_dag[2, 1], "block_11": U_dag[2, 2],
    }
    return _update_u_info_recursive(ms, u_info)


def make_model() -> LookupState:
    return LookupState(N=N, total_sz=TOTAL_SZ, param_dtype=PARAM_DTYPE, init_stddev=INIT_STDDEV)


def make_vstate(model, *, seed: int) -> nk.vqs.MCState:
    """MCState with the nearest-neighbour exchange sampler (stays in the Sz sector)."""
    sampler = nk.sampler.MetropolisExchange(
        hilbert=HILBERT, n_chains=N_CHAINS, graph=GRAPH, sweep_size=THIN_COEFF * N)
    return nk.vqs.MCState(
        sampler=sampler, model=model, n_samples=N_SAMPLES, n_discard_per_chain=N_DISCARD,
        seed=seed, chunk_size=max(1, N_CHAINS // jax.local_device_count()),
    )


def make_solver():
    if SOLVER == "cholesky":
        return nk.optimizer.solver.cholesky
    if SOLVER == "pinv_smooth":
        return nk.optimizer.solver.pinv_smooth
    if SOLVER == "cg":
        return partial(jax.scipy.sparse.linalg.cg, tol=1e-5, maxiter=500)
    raise ValueError(f"unknown solver {SOLVER!r} (use 'cholesky', 'pinv_smooth' or 'cg')")


def make_sr_driver(hamiltonian, vstate) -> nk.driver.VMC_SR:
    """Standard NetKet VMC-SR: clipped SGD step along the SR direction (real parameters)."""
    optimizer = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.sgd(learning_rate=LR))
    use_ntk = vstate.n_parameters > vstate.n_samples
    return nk.driver.VMC_SR(
        hamiltonian=hamiltonian, optimizer=optimizer, variational_state=vstate,
        diag_shift=DIAG_SHIFT, mode="real", linear_solver=make_solver(), use_ntk=use_ntk,
    )


def thermalize(vstate: nk.vqs.MCState, n_discard: int = 20) -> None:
    vstate.n_discard_per_chain = n_discard
    vstate.reset()
    _ = vstate.samples
    vstate.n_discard_per_chain = N_DISCARD


def run_config(**extra) -> dict:
    cfg = {
        "N": N, "lx": LX, "pbc": PBC, "J1": J1, "J2": J2, "total_sz": TOTAL_SZ,
        "n_samples": N_SAMPLES, "n_chains": N_CHAINS, "n_discard": N_DISCARD, "thin_coeff": THIN_COEFF,
        "lr": LR, "diag_shift": DIAG_SHIFT, "solver": SOLVER, "grad_clip": GRAD_CLIP, "seed": SEED,
        "config_path": CONFIG_PATH,
    }
    cfg.update(extra)
    return cfg


def save_ckpt(fmt: str, sweep: int, vstate, history: list[dict], **extra) -> str:
    """Pickle {parameters, sampler_state, history, config} of the *un-rotated* state."""
    path = fmt.format(sweep)
    save_checkpoint(path, vstate, list(history), run_config(sweep=int(sweep), **extra))
    return path


def restore_into(vstate, ckpt: dict) -> None:
    """Load checkpoint parameters (and the sampler state when its shape matches) into ``vstate``."""
    template = vstate.parameters
    loaded = ckpt["parameters"]
    if jax.tree_util.tree_structure(loaded) != jax.tree_util.tree_structure(template):
        raise ValueError("checkpoint parameter tree does not match the current ansatz")
    vstate.parameters = jax.tree_util.tree_map(lambda v, t: jnp.asarray(v, dtype=t.dtype), loaded, template)
    saved = ckpt.get("sampler_state")
    if saved is not None and saved.σ.shape == vstate.sampler_state.σ.shape:
        vstate.sampler_state = saved
        print(f"  restored sampler state ({saved.σ.shape})")
    else:
        print("  sampler state not restored (missing or different chain layout): thermalizing")
        thermalize(vstate)


def load_ckpt(path: str) -> dict:
    return load_checkpoint(path)


def record(sweep: int, stats) -> dict:
    """One history entry from NetKet ``Stats`` (energy of the table at ``sweep``)."""
    return {
        "sweep": int(sweep),
        "energy": float(np.real(stats.mean)),
        "energy_err": float(np.real(stats.error_of_mean)),
    }


def history_from_log(log: nk.logging.RuntimeLog) -> list[dict]:
    """History entries of the ``Energy`` trace collected so far by a NetKet logger."""
    if "Energy" not in log.data:
        return []
    e = log.data["Energy"]
    return [
        {"sweep": int(it), "energy": float(np.real(m)), "energy_err": float(np.real(s))}
        for it, m, s in zip(e.iters, e["Mean"], e["Sigma"])
    ]


def save_history(history: list[dict], path: str, **extra) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        sweeps=np.array([h["sweep"] for h in history]),
        energies=np.array([h["energy"] for h in history]),
        energy_errs=np.array([h["energy_err"] for h in history]),
        **extra,
    )


def load_ed_energy() -> float | None:
    """Ground-state energy from ed_ground_state.py, if it has been run."""
    if not os.path.exists(ED_GS_PATH):
        return None
    with np.load(ED_GS_PATH) as d:
        return float(np.min(d["eigvals"]))


def banner(title: str) -> None:
    print("=" * 70)
    print(f"{title}  {LX}x{LX} {'PBC' if PBC else 'OBC'} J1-J2 Heisenberg  J1={J1} J2={J2}  "
          f"lr={LR} shift={DIAG_SHIFT} solver={SOLVER}")
    print("=" * 70)
    print(f"  Backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(f"  Config:  {CONFIG_PATH}")
    print(f"  Bonds:   {len(NN_BONDS)} NN + {len(NNN_BONDS)} NNN")
    e_ref = load_ed_energy()
    if e_ref is not None:
        print(f"  ED reference: E_gs = {e_ref:.8f}  (E/N = {e_ref / N:.8f})")
