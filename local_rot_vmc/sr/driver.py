"""Minimal holomorphic RK4 / adaptive RK45 (Dormand-Prince) SR/TDVP driver with explicit timing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
import numpy as np
import jax.numpy as jnp
import netket.jax as nkjax
from netket import stats as nkstats
from netket.operator import AbstractOperator
from netket.optimizer.solver import cholesky as nk_cholesky
from netket.vqs import MCState

from local_rot_vmc.sr.integrators import (  # noqa: F401
    AdaptiveRK45, FixedRK4, StepResult, dormand_prince_trial, make_integrator, qgt_norm, scaled_l2_norm,
    tree_all_finite,
)


def _set_u_recursive(tree, u_info):
    result = dict(tree)
    for key, value in u_info.items():
        if key in result:
            result[key] = value
    for key, val in result.items():
        if isinstance(val, dict):
            result[key] = _set_u_recursive(val, u_info)
    return result


def _update_u_info_recursive(model_state, u_info):
    ms = dict(model_state)
    if 'u_info' in ms:
        ms['u_info'] = _set_u_recursive(ms['u_info'], u_info)
    return ms


def _set_ij_recursive(tree, bond_info):
    result = dict(tree)
    result['i'] = bond_info['i']
    result['j'] = bond_info['j']
    for key, val in result.items():
        if key not in ('i', 'j') and isinstance(val, dict):
            result[key] = _set_ij_recursive(val, bond_info)
    return result


def _update_bond_info_recursive(model_state, bond_info):
    ms = dict(model_state)
    if 'bond_info' in ms:
        ms['bond_info'] = _set_ij_recursive(ms['bond_info'], bond_info)
    return ms


def _set_pos_recursive(tree, site_info):
    result = dict(tree)
    result['pos'] = site_info['pos']
    for key, val in result.items():
        if key != 'pos' and isinstance(val, dict):
            result[key] = _set_pos_recursive(val, site_info)
    return result


def _update_site_info_recursive(model_state, site_info):
    ms = dict(model_state)
    if 'site_info' in ms:
        ms['site_info'] = _set_pos_recursive(ms['site_info'], site_info)
    return ms


def update_model_state_bond(model_state, i, j):
    """Update bond info in a model_state dict. Works without a driver."""
    bond_info = {
        'i': jnp.array(i, dtype=jnp.int32),
        'j': jnp.array(j, dtype=jnp.int32),
    }
    return _update_bond_info_recursive(model_state, bond_info)


def update_model_state_site(model_state, pos):
    """Update site info in a model_state dict. Works without a driver."""
    site_info = {
        'pos': jnp.array(pos, dtype=jnp.int32),
    }
    return _update_site_info_recursive(model_state, site_info)


def update_model_state_u(model_state, U):
    """Update 2x2 or 4x4 U† components in a model_state dict."""
    U = jnp.asarray(U)
    U_dag = U.conj().T

    if U.shape == (4, 4):
        u_info = {
            'phi1': jnp.array(U_dag[0, 0], dtype=jnp.complex128),
            'phi2': jnp.array(U_dag[3, 3], dtype=jnp.complex128),
            'block_00': jnp.array(U_dag[1, 1], dtype=jnp.complex128),
            'block_01': jnp.array(U_dag[1, 2], dtype=jnp.complex128),
            'block_10': jnp.array(U_dag[2, 1], dtype=jnp.complex128),
            'block_11': jnp.array(U_dag[2, 2], dtype=jnp.complex128),
        }
    elif U.shape == (2, 2):
        u_info = {
            'u_00': jnp.array(U_dag[0, 0], dtype=jnp.complex128),
            'u_01': jnp.array(U_dag[0, 1], dtype=jnp.complex128),
            'u_10': jnp.array(U_dag[1, 0], dtype=jnp.complex128),
            'u_11': jnp.array(U_dag[1, 1], dtype=jnp.complex128),
        }
    else:
        raise ValueError(f"U must be 2x2 or 4x4, got shape {U.shape}")

    return _update_u_info_recursive(model_state, u_info)


def update_model_state_bond_and_u(model_state, bond, U):
    """Update both bond and U† in a model_state dict. Works without a driver."""
    ms = update_model_state_bond(model_state, bond[0], bond[1])
    ms = update_model_state_u(ms, U)
    return ms


def update_model_state_site_and_u(model_state, pos, U):
    """Update both site and U† in a model_state dict. Works without a driver."""
    ms = update_model_state_site(model_state, pos)
    ms = update_model_state_u(ms, U)
    return ms


KernelKind = Literal["qgt", "ntk"]
PropagationKind = Literal["real", "imag"]


@dataclass
class _KernelResult:
    dp: Any
    loss_stats: Any
    energy_mean: complex
    directional_mean: complex
    residual_norm: float


class HolomorphicTDVP:
    """Minimal holomorphic TDVP driver (fixed-step RK4 or adaptive RK45)."""

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        *,
        lr: float,
        variational_state: MCState,
        diag_shift: float,
        propagation_type: PropagationKind = "real",
        kernel: KernelKind = "qgt",
        linear_solver: Any = nk_cholesky,
        chunk_size_bwd: int | None = None,
        holomorphic: bool = True,
        mode: str | None = None,
        integrator="rk4",
        rtol: float = 1e-4,
        atol: float = 1e-8,
        dt_min: float = 1e-5,
        dt_max: float = 1e-2,
        sample_policy: str = "fresh",
        error_norm: str = "l2",
    ):
        """``integrator``: rk4 | rk45 | Integrator; ``sample_policy``: fresh | cached; ``error_norm``: l2 | qgt."""
        if not isinstance(variational_state, MCState):
            raise TypeError("HolomorphicTDVP only supports netket.vqs.MCState.")
        if not holomorphic:
            raise ValueError("HolomorphicTDVP only supports holomorphic=True.")
        if mode not in (None, "holomorphic"):
            raise ValueError("HolomorphicTDVP only supports mode=None or mode='holomorphic'.")
        if propagation_type not in ("real", "imag"):
            raise ValueError("propagation_type must be 'real' or 'imag'.")
        if kernel not in ("qgt", "ntk"):
            raise ValueError("kernel must be 'qgt' or 'ntk'.")
        if sample_policy not in ("fresh", "cached"):
            raise ValueError("sample_policy must be 'fresh' or 'cached'.")
        if error_norm not in ("l2", "qgt"):
            raise ValueError("error_norm must be 'l2' or 'qgt'.")
        if error_norm == "qgt" and kernel != "qgt":
            raise ValueError("error_norm='qgt' needs kernel='qgt'.")
        self._solver = linear_solver
        self._integrator = make_integrator(integrator, rtol=rtol, atol=atol, dt_min=dt_min, dt_max=dt_max)
        self.integrator = "rk45" if self._integrator.adaptive else "rk4"
        self.rtol, self.atol = float(rtol), float(atol)
        self.dt_min, self.dt_max = float(dt_min), float(dt_max)
        self.sample_policy = sample_policy
        self.error_norm = error_norm

        params = jax.tree_util.tree_leaves(variational_state.parameters)
        if not params:
            raise ValueError("Variational state has no parameters.")
        if not all(jnp.iscomplexobj(x) for x in params):
            raise ValueError("HolomorphicTDVP requires complex holomorphic parameters.")

        self.state = variational_state
        self._ham = hamiltonian
        self._lr = float(lr)
        self.diag_shift = float(diag_shift)
        self.kernel = kernel
        self.propagation_type = propagation_type
        self.chunk_size_bwd = chunk_size_bwd or variational_state.chunk_size
        self.mode = "holomorphic"

        self.step_count = 0
        self._loss_stats = None
        self.last_residual_norm: float = 0.0
        self.t: float = 0.0
        self.dt: float = self._lr
        self.last_dt: float = 0.0
        self.last_error: float = 0.0
        self.n_rejected: int = 0
        self.n_rhs_evals: int = 0
        self._k_fsal = None
        self._cache = None
        self._last_S = None

    def run(self, n_steps: int, show_progress: bool = False):
        """Advance by ``n_steps`` accepted steps (rk4: n_steps * lr in time; rk45: adaptive)."""
        del show_progress
        for _ in range(n_steps):
            self._step()

    def run_until(self, t_max: float, max_steps: int | None = None) -> int:
        """Advance until ``self.t >= t_max`` (rk45 clips the last step to land on t_max)."""
        n = 0
        while self.t < t_max - 1e-12 and (max_steps is None or n < max_steps):
            if self._integrator.adaptive:
                self.dt = min(self.dt, t_max - self.t)
            self.run(1)
            n += 1
        return n

    def _gamma(self) -> complex:
        return -1.0j if self.propagation_type == "real" else -1.0

    def _flatten_samples(self, samples):
        if samples.ndim > 2:
            return jax.lax.collapse(samples, 0, samples.ndim - 1)
        return samples

    def begin_step(self):
        """Called before every integrator step."""
        self._last_S = None
        if self.sample_policy != "cached":
            return
        self.state.reset()
        samples = self._flatten_samples(self.state.samples)
        logpsi0 = self._log_psi(samples)
        self._cache = {"samples": samples, "logpsi0": logpsi0}

    def end_step(self):
        self._cache = None

    def _log_psi(self, samples):
        return self.state.log_value(samples)

    def _local_energies_on(self, samples, logpsi):
        sigma_p, mels = self._ham.get_conn_padded(np.asarray(samples))
        sigma_p = jnp.asarray(sigma_p)
        logpsi_p = self._log_psi(sigma_p.reshape(-1, sigma_p.shape[-1])).reshape(sigma_p.shape[:-1])
        return jnp.sum(jnp.asarray(mels) * jnp.exp(logpsi_p - logpsi[:, None]), axis=1)

    def _collect_local_data(self):
        if self.sample_policy == "cached" and self._cache is not None:
            samples = self._cache["samples"]
            logpsi = self._log_psi(samples)
            logw = 2.0 * (logpsi - self._cache["logpsi0"]).real
            w = jnp.exp(logw - jnp.max(logw))
            weights = w / jnp.sum(w)
            local_energies = self._local_energies_on(samples, logpsi)
            loss_stats = nkstats.statistics(local_energies)
            energy_mean = jnp.sum(weights * local_energies)
            return samples, local_energies, weights, loss_stats, energy_mean
        self.state.reset()
        local_energies = self.state.local_estimators(self._ham)
        samples = self.state.samples
        if local_energies.ndim > 1:
            local_energies = jax.lax.collapse(local_energies, 0, local_energies.ndim)
        samples = self._flatten_samples(samples)
        loss_stats = nkstats.statistics(local_energies)
        energy_mean = jnp.mean(local_energies)
        weights = jnp.full((samples.shape[0],), 1.0 / samples.shape[0], dtype=local_energies.real.dtype)
        return samples, local_energies, weights, loss_stats, energy_mean

    def _compute_log_jacobian(self, samples):
        return nkjax.jacobian(
            self.state._apply_fun,
            self.state.parameters,
            samples,
            self.state.model_state,
            mode="holomorphic",
            dense=True,
            center=False,
            chunk_size=self.chunk_size_bwd,
            _axis_0_is_sharded=True,
        )

    def _compute_update_qgt(self) -> _KernelResult:
        samples, local_energies, weights, loss_stats, energy_mean = self._collect_local_data()
        O = self._compute_log_jacobian(samples)

        gamma = self._gamma()
        diag_shift = self.diag_shift
        solver = self._solver

        @jax.jit
        def _qgt_solve(O, local_energies, weights):
            Ow = O * weights[:, None]
            S = Ow.conj().T @ O
            F = Ow.conj().T @ local_energies
            idx = jnp.arange(S.shape[0])
            S_shifted = S.at[idx, idx].add(diag_shift)
            rhs = gamma * F

            dp, _ = solver(S_shifted, rhs)

            residual_norm = jnp.linalg.norm(S @ dp - rhs) / jnp.linalg.norm(rhs)
            directional_mean = jnp.sum(weights * (O @ dp))
            return dp, directional_mean, residual_norm, S

        dp, directional_mean, residual_norm, S = _qgt_solve(O, local_energies, weights)
        if self._last_S is None:
            self._last_S = S
        return _KernelResult(
            dp=dp,
            loss_stats=loss_stats,
            energy_mean=complex(energy_mean),
            directional_mean=complex(directional_mean),
            residual_norm=float(residual_norm),
        )

    def _compute_update_ntk(self) -> _KernelResult:
        samples, local_energies, weights, loss_stats, energy_mean = self._collect_local_data()
        O = self._compute_log_jacobian(samples)

        gamma = self._gamma()
        diag_shift = self.diag_shift
        solver = self._solver

        @jax.jit
        def _ntk_solve(O, local_energies, weights):
            sw = jnp.sqrt(weights)
            Ow = O * sw[:, None]
            ew = sw * local_energies
            K = Ow @ Ow.conj().T
            idx = jnp.arange(K.shape[0])
            K_shifted = K.at[idx, idx].add(diag_shift)
            rhs = gamma * ew

            alpha, _ = solver(K_shifted, rhs)

            dp = Ow.conj().T @ alpha
            directional_mean = jnp.sum(weights * (O @ dp))
            Sx = Ow.conj().T @ (Ow @ dp)
            F_param = gamma * (Ow.conj().T @ ew)
            residual_norm = jnp.linalg.norm(Sx - F_param) / jnp.linalg.norm(F_param)
            return dp, directional_mean, residual_norm

        dp, directional_mean, residual_norm = _ntk_solve(O, local_energies, weights)
        return _KernelResult(
            dp=dp,
            loss_stats=loss_stats,
            energy_mean=complex(energy_mean),
            directional_mean=complex(directional_mean),
            residual_norm=float(residual_norm),
        )

    def _compute_update(self) -> _KernelResult:
        if self.kernel == "qgt":
            return self._compute_update_qgt()
        else:
            return self._compute_update_ntk()

    def _unflatten_like_params(self, flat_update):
        flat_update = jnp.asarray(flat_update)
        _, unravel = jax.flatten_util.ravel_pytree(self.state.parameters)
        return unravel(flat_update)

    def _rk4_substep(self):
        result = self._compute_update()
        self._loss_stats = result.loss_stats
        self.last_residual_norm = result.residual_norm
        self.n_rhs_evals += 1
        return self._unflatten_like_params(result.dp)

    def _rhs(self, theta):
        self._update_parameters(theta)
        return self._rk4_substep()

    def _update_parameters(self, params):
        self.state.parameters = params

    def _error_norm(self):
        if self.error_norm == "qgt":
            return qgt_norm(self.rtol, self.atol, lambda: self._last_S)
        return scaled_l2_norm(self.rtol, self.atol)

    def _step(self):
        theta0 = self.state.parameters
        integ = self._integrator
        self.begin_step()
        k1 = self._k_fsal if (integ.adaptive and self.sample_policy == "fresh") else None
        while True:
            dt = self.dt if integ.adaptive else self._lr
            res = integ.step(theta0, dt, self._rhs, norm=self._error_norm(), k1=k1)
            if res.accepted:
                self._update_parameters(res.theta)
                self._k_fsal = res.k_last
                self.t += res.dt_used
                self.last_dt, self.last_error = res.dt_used, res.error
                if integ.adaptive:
                    self.dt = res.dt_next
                self.step_count += 1
                self.end_step()
                return
            self.n_rejected += 1
            if not np.isfinite(res.error) and dt <= integ.dt_min * (1 + 1e-9):
                self._update_parameters(theta0)
                self.end_step()
                raise RuntimeError(f"RK45: non-finite update at dt_min = {integ.dt_min:g} (t = {self.t:.6f})")
            self.dt = res.dt_next
            k1 = None if not np.isfinite(res.error) else k1

    def _rk4_step(self):
        self._step()

    def _rk45_step(self):
        self._step()

    def reset_fsal(self):
        """Forget the cached first stage and sample cache (call after changing parameters/state from outside)."""
        self._k_fsal = None
        self._cache = None

    def set_active_site(self, pos: int):
        self._k_fsal = None
        """Update the model's active site index. No JIT retrace."""
        self.state.model_state = update_model_state_site(
            self.state.model_state, pos
        )

    def set_active_site_and_u(self, pos: int, U):
        """Update both site and U in one call. No JIT retrace."""
        self.state.model_state = update_model_state_site_and_u(
            self.state.model_state, pos, U
        )

