"""Time integrators for the holomorphic TDVP driver (plugin interface)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np

Pytree = Any
RHS = Callable[[Pytree], Pytree]
ErrorNorm = Callable[[Pytree, Pytree, Pytree], float]


@dataclass
class StepResult:
    theta: Pytree
    dt_used: float
    dt_next: float
    accepted: bool
    error: float
    n_evals: int
    k_last: Pytree | None = None


class Integrator(Protocol):
    adaptive: bool

    def step(self, theta: Pytree, dt: float, rhs: RHS, *, norm: ErrorNorm | None = None,
             k1: Pytree | None = None) -> StepResult: ...


def axpy_tree(theta0: Pytree, dt: float, coeffs, ks) -> Pytree:
    """theta0 + dt * sum_j coeffs[j] * ks[j] on pytrees."""
    out = theta0
    for c, k in zip(coeffs, ks):
        if c != 0.0:
            out = jax.tree_util.tree_map(lambda p, a, c=c: p + dt * c * a, out, k)
    return out


def tree_all_finite(tree: Pytree) -> bool:
    return all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(tree))


def scaled_l2_norm(rtol: float, atol: float) -> ErrorNorm:
    """Component-wise scaled RMS error, sqrt(mean |err / (atol + rtol * max(|theta0|, |theta1|))|^2)."""
    def norm(theta0, theta1, err):
        num, n = 0.0, 0
        for a, b, e in zip(jax.tree_util.tree_leaves(theta0), jax.tree_util.tree_leaves(theta1),
                           jax.tree_util.tree_leaves(err)):
            sc = atol + rtol * jnp.maximum(jnp.abs(a), jnp.abs(b))
            num += float(jnp.sum(jnp.abs(e / sc) ** 2))
            n += e.size
        return float(np.sqrt(num / n))
    return norm


def qgt_norm(rtol: float, atol: float, get_S: Callable[[], jax.Array]) -> ErrorNorm:
    """QGT-induced error norm ||err||_S / (atol + rtol * ||theta0||_S)."""
    def norm(theta0, theta1, err):
        S = get_S()
        v0 = jax.flatten_util.ravel_pytree(theta0)[0]
        e = jax.flatten_util.ravel_pytree(err)[0]
        n_e = float(jnp.sqrt(jnp.abs(jnp.vdot(e, S @ e))))
        n_0 = float(jnp.sqrt(jnp.abs(jnp.vdot(v0, S @ v0))))
        return n_e / (atol + rtol * n_0)
    return norm


class FixedRK4:
    """Classical RK4, always accepts, dt_next = dt."""
    adaptive = False

    def step(self, theta, dt, rhs, *, norm=None, k1=None) -> StepResult:
        k1 = rhs(theta) if k1 is None else k1
        k2 = rhs(axpy_tree(theta, dt, (0.5,), (k1,)))
        k3 = rhs(axpy_tree(theta, dt, (0.5,), (k2,)))
        k4 = rhs(axpy_tree(theta, dt, (1.0,), (k3,)))
        theta_new = jax.tree_util.tree_map(
            lambda p, a, b, c, d: p + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d), theta, k1, k2, k3, k4)
        return StepResult(theta_new, dt, dt, True, 0.0, 4)


_DP_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_DP_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_DP_B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)


def dormand_prince_trial(theta0, dt, f, k1=None):
    """One Dormand-Prince 5(4) trial step for dtheta/dt = f(theta)."""
    ks = [k1 if k1 is not None else f(theta0)]
    for i in range(1, 7):
        ks.append(f(axpy_tree(theta0, dt, _DP_A[i], ks)))
    theta5 = axpy_tree(theta0, dt, _DP_B5, ks)
    theta4 = axpy_tree(theta0, dt, _DP_B4, ks)
    err = jax.tree_util.tree_map(lambda a, b: a - b, theta5, theta4)
    return theta5, err, ks[6]


class AdaptiveRK45:
    """Embedded Dormand-Prince 5(4) with the standard step-size controller."""
    adaptive = True

    def __init__(self, *, rtol: float = 1e-4, atol: float = 1e-8, dt_min: float = 1e-5, dt_max: float = 1e-2,
                 safety: float = 0.9, fac_min: float = 0.2, fac_max: float = 5.0):
        self.rtol, self.atol = float(rtol), float(atol)
        self.dt_min, self.dt_max = float(dt_min), float(dt_max)
        self.safety, self.fac_min, self.fac_max = float(safety), float(fac_min), float(fac_max)

    def default_norm(self) -> ErrorNorm:
        return scaled_l2_norm(self.rtol, self.atol)

    def clip_dt(self, dt: float) -> float:
        return float(min(self.dt_max, max(self.dt_min, dt)))

    def step(self, theta, dt, rhs, *, norm=None, k1=None) -> StepResult:
        norm = norm or self.default_norm()
        theta5, err, k7 = dormand_prince_trial(theta, dt, rhs, k1)
        n_evals = 7 if k1 is None else 6
        if not tree_all_finite(theta5):
            return StepResult(theta, dt, self.clip_dt(dt / 5.0), False, float("inf"), n_evals)
        error = norm(theta, theta5, err)
        fac = self.safety * error ** (-0.2) if error > 0 else self.fac_max
        dt_next = self.clip_dt(dt * min(self.fac_max, max(self.fac_min, fac)))
        accepted = error <= 1.0 or dt <= self.dt_min * (1 + 1e-9)
        return StepResult(theta5, dt, dt_next, accepted, error, n_evals, k_last=k7)


def make_integrator(spec, **kwargs):
    """``"rk4"`` / ``"rk45"`` (with rtol, atol, dt_min, dt_max kwargs) or an Integrator instance."""
    if isinstance(spec, str):
        if spec == "rk4":
            return FixedRK4()
        if spec == "rk45":
            return AdaptiveRK45(**kwargs)
        raise ValueError(f"unknown integrator {spec!r}; use 'rk4', 'rk45' or an Integrator instance")
    return spec
