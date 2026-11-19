"""Optimized RotatedModel for Sz-conserving Local-Rotation VMC."""

import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple
from jax.scipy.special import logsumexp

from local_rot_vmc.operators.unitary import batch_two_spin_config_to_index


REAL_DTYPE = jnp.float64
EPS_IN_LOG = jnp.array(1e-300, dtype=REAL_DTYPE)
LOG_EPS = jnp.log(EPS_IN_LOG)


def _batch_single_spin_config_to_index(x: jnp.ndarray, pos: int) -> jnp.ndarray:
    return ((x[:, pos] + 1) // 2).astype(jnp.int32)


def _is_sz_conserving(U: jnp.ndarray, tol: float = 1e-6) -> bool:
    off_diagonal_indices = [
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 3),
        (2, 0), (2, 3),
        (3, 0), (3, 1), (3, 2),
    ]

    for i, j in off_diagonal_indices:
        if abs(U[i, j]) > tol:
            return False

    return True


def create_sz_conserving_rotated_model(
    base_model: nn.Module,
    U: jnp.ndarray,
    bond: Tuple[int, int],
) -> nn.Module:
    """Create an optimized Flax model with Local-Rotation baked in."""
    if not _is_sz_conserving(U):
        raise ValueError(
            f"U matrix is not Sz-conserving (not block-diagonal). "
            f"create_sz_conserving_rotated_model requires conserve_sz=True. "
            f"U[0,1]={U[0,1]:.2e}, U[1,0]={U[1,0]:.2e} should be near zero."
        )

    i, j = bond
    U_dag = U.conj().T

    phi1 = U_dag[0, 0]
    phi2 = U_dag[3, 3]
    block_00 = U_dag[1, 1]
    block_01 = U_dag[1, 2]
    block_10 = U_dag[2, 1]
    block_11 = U_dag[2, 2]

    base_apply = base_model.apply

    class OptimizedRotatedModel(nn.Module):
        def setup(self):
            self.base = base_model

        def __call__(self, x):
            return jnp.log(self.get_psi_val(x).astype(jnp.complex128) + EPS_IN_LOG)

        def get_psi_val(self, x):
            batch_size = x.shape[0]
            idx_sigma = batch_two_spin_config_to_index(x, i, j)

            x_b0 = x.at[:, i].set(1).at[:, j].set(1)
            x_b1 = x.at[:, i].set(1).at[:, j].set(-1)
            x_b2 = x.at[:, i].set(-1).at[:, j].set(1)
            x_b3 = x.at[:, i].set(-1).at[:, j].set(-1)

            if hasattr(self.base, 'get_psi_val'):
                x_all = jnp.concatenate([x_b0, x_b1, x_b2, x_b3], axis=0)
                psi_all = self.base.get_psi_val(x_all)
                psi0, psi1, psi2, psi3 = jnp.split(psi_all, 4, axis=0)
            else:
                raise NotImplementedError("Base model does not have get_psi_val")

            r0 = phi1 * psi0
            r1 = block_00 * psi1 + block_01 * psi2
            r2 = block_10 * psi1 + block_11 * psi2
            r3 = phi2 * psi3

            mask_0 = (idx_sigma == 0)
            mask_1 = (idx_sigma == 1)
            mask_2 = (idx_sigma == 2)

            result = jnp.where(mask_0, r0,
                       jnp.where(mask_1, r1,
                       jnp.where(mask_2, r2, r3)))
            return result

    return OptimizedRotatedModel()




def create_sz_projected_model(
    base_model: nn.Module,
    total_sz: float,
) -> nn.Module:
    """Wrap a base model with Sz projection."""
    class SzProjectedModel(nn.Module):
        def setup(self):
            self.base = base_model

        def get_psi_val(self, x):
            sz = jnp.sum(x, axis=-1) / 2.0
            sz_ok = jnp.abs(sz - total_sz) < 0.1

            if hasattr(self.base, "get_psi_val") and callable(getattr(self.base, "get_psi_val")):
                psi = self.base.get_psi_val(x)
            else:
                psi = jnp.exp(self.base(x))

            return jnp.where(
                sz_ok,
                psi,
                jnp.zeros_like(psi, dtype=jnp.complex128),
            )

        def __call__(self, x):
            psi = self.get_psi_val(x)
            return jnp.log(psi.astype(jnp.complex128) + EPS_IN_LOG)

    return SzProjectedModel()


def create_dynamic_u_sz_conserving_rotated_model(
    base_model: nn.Module,
) -> nn.Module:
    """Create a dynamic rotated model with MUTABLE U components."""

    class DynamicMutableUSzConservingRotatedModel(nn.Module):
        def setup(self):
            self.base = base_model
            self.i_var = self.variable('bond_info', 'i', lambda: jnp.array(0, dtype=jnp.int32))
            self.j_var = self.variable('bond_info', 'j', lambda: jnp.array(1, dtype=jnp.int32))
            self.phi1_var = self.variable('u_info', 'phi1', lambda: jnp.array(1.0 + 0j, dtype=jnp.complex128))
            self.phi2_var = self.variable('u_info', 'phi2', lambda: jnp.array(1.0 + 0j, dtype=jnp.complex128))
            self.block_00_var = self.variable('u_info', 'block_00', lambda: jnp.array(1.0 + 0j, dtype=jnp.complex128))
            self.block_01_var = self.variable('u_info', 'block_01', lambda: jnp.array(0.0 + 0j, dtype=jnp.complex128))
            self.block_10_var = self.variable('u_info', 'block_10', lambda: jnp.array(0.0 + 0j, dtype=jnp.complex128))
            self.block_11_var = self.variable('u_info', 'block_11', lambda: jnp.array(1.0 + 0j, dtype=jnp.complex128))

        def __call__(self, x):
            return jnp.log(self.get_psi_val(x).astype(jnp.complex128) + EPS_IN_LOG)

        def get_psi_val(self, x):
            i = self.i_var.value
            j = self.j_var.value
            batch_size = x.shape[0]
            idx_sigma = batch_two_spin_config_to_index(x, i, j)

            x_b0 = x.at[:, i].set(1).at[:, j].set(1)
            x_b1 = x.at[:, i].set(1).at[:, j].set(-1)
            x_b2 = x.at[:, i].set(-1).at[:, j].set(1)
            x_b3 = x.at[:, i].set(-1).at[:, j].set(-1)

            x_config1 = jnp.where((idx_sigma == 0)[:, None], x_b0, x_b1)
            x_config1 = jnp.where((idx_sigma == 3)[:, None], x_b3, x_config1)
            x_config2 = jnp.where(((idx_sigma == 1) | (idx_sigma == 2))[:, None], x_b2, x_config1)

            x_batch = jnp.concatenate([x_config1, x_config2], axis=0)
            if hasattr(self.base, 'get_psi_val'):
                psi_batch = self.base.get_psi_val(x_batch)
            else:
                logpsi_batch = self.base(x_batch)
                psi_batch = jnp.exp(logpsi_batch)

            psi1 = psi_batch[:batch_size]
            psi2 = psi_batch[batch_size:]

            _phi1 = self.phi1_var.value
            _phi2 = self.phi2_var.value
            _block_00 = self.block_00_var.value
            _block_01 = self.block_01_var.value
            _block_10 = self.block_10_var.value
            _block_11 = self.block_11_var.value

            r0 = _phi1 * (psi1 + psi2) * 0.5
            r1 = _block_00 * psi1 + _block_01 * psi2
            r2 = _block_10 * psi1 + _block_11 * psi2
            r3 = _phi2 * (psi1 + psi2) * 0.5

            mask_0 = (idx_sigma == 0)
            mask_1 = (idx_sigma == 1)
            mask_2 = (idx_sigma == 2)

            return jnp.where(mask_0, r0,
                       jnp.where(mask_1, r1,
                       jnp.where(mask_2, r2, r3)))

    return DynamicMutableUSzConservingRotatedModel()


def create_dynamic_bond_projected_model(
    base_model: nn.Module,
    fixed: bool = True,
) -> nn.Module:
    """Wrap a base model with dynamic bond projection P[s_i, s_j]."""
    _fixed = fixed

    class DynamicBondProjectedModel(nn.Module):
        def setup(self):
            self.base = base_model
            self.i_var = self.variable('bond_info', 'i', lambda: jnp.array(0, dtype=jnp.int32))
            self.j_var = self.variable('bond_info', 'j', lambda: jnp.array(1, dtype=jnp.int32))
            if not _fixed:
                self.log_P_param = self.param('log_P',
                    lambda rng: jnp.array([0.0, 0.0, 0.0, LOG_EPS]))

        def __call__(self, x):
            i = self.i_var.value
            j = self.j_var.value
            log_psi = self.base(x)

            si = ((1 - x[:, i]) // 2).astype(jnp.int32)
            sj = ((1 - x[:, j]) // 2).astype(jnp.int32)
            idx = si * 2 + sj

            if _fixed:
                log_P = jnp.array([0.0, 0.0, 0.0, LOG_EPS])
            else:
                log_P = self.log_P_param

            return log_P[idx] + log_psi

        def get_psi_val(self, x):
            if not hasattr(self.base, 'get_psi_val'):
                raise NotImplementedError("Base model does not have get_psi_val")
            psi = self.base.get_psi_val(x)
            i = self.i_var.value
            j = self.j_var.value
            si = ((1 - x[:, i]) // 2).astype(jnp.int32)
            sj = ((1 - x[:, j]) // 2).astype(jnp.int32)
            idx = si * 2 + sj
            if _fixed:
                log_P = jnp.array([0.0, 0.0, 0.0, LOG_EPS])
            else:
                log_P = self.log_P_param
            return jnp.exp(log_P[idx]) * psi

    return DynamicBondProjectedModel()


def create_dynamic_1site_rotated_model(
    base_model: nn.Module,
    U: jnp.ndarray,
    helper_model_sz_constraint: Optional[float] = None,
    helper_model_coeff: float = 0.0,
) -> nn.Module:
    """Create a dynamic rotated model for a single-site 2x2 unitary."""
    U = jnp.asarray(U)
    if U.shape != (2, 2):
        raise ValueError(f"U must be 2x2 for single-site rotation, got shape {U.shape}")

    _U_dag = U.conj().T
    _helper_sz = helper_model_sz_constraint
    _helper_coeff = helper_model_coeff
    _use_helper = _helper_sz is not None
    _amplitude_space = _use_helper

    class Dynamic1SiteRotatedModel(nn.Module):
        def setup(self):
            self.base = base_model
            self.pos_var = self.variable(
                'site_info', 'pos', lambda: jnp.array(0, dtype=jnp.int32)
            )
            self.u_00_var = self.variable(
                'u_info', 'u_00', lambda: jnp.asarray(_U_dag[0, 0], dtype=jnp.complex128)
            )
            self.u_01_var = self.variable(
                'u_info', 'u_01', lambda: jnp.asarray(_U_dag[0, 1], dtype=jnp.complex128)
            )
            self.u_10_var = self.variable(
                'u_info', 'u_10', lambda: jnp.asarray(_U_dag[1, 0], dtype=jnp.complex128)
            )
            self.u_11_var = self.variable(
                'u_info', 'u_11', lambda: jnp.asarray(_U_dag[1, 1], dtype=jnp.complex128)
            )

        def _build_basis_configs(self, x):
            pos = self.pos_var.value
            x_expanded = jnp.tile(x[:, None, :], (1, 2, 1))
            x_expanded = x_expanded.at[:, 0, pos].set(-1)
            x_expanded = x_expanded.at[:, 1, pos].set(1)
            return x_expanded

        def _u_dag(self):
            return jnp.asarray(
                [
                    [self.u_00_var.value, self.u_01_var.value],
                    [self.u_10_var.value, self.u_11_var.value],
                ],
                dtype=jnp.complex128,
            )

        def _eval_base_psi(self, x):
            if hasattr(self.base, "get_psi_val"):
                return self.base.get_psi_val(x)
            return jnp.exp(self.base(x))

        def _helper_activation_mask(self, x_basis):
            if not _use_helper:
                return jnp.zeros((x_basis.shape[0],), dtype=bool)

            pos = self.pos_var.value
            twice_sz = jnp.sum(x_basis, axis=-1)
            in_helper_sector = (twice_sz == int(2 * _helper_sz))
            spin_up_at_pos = (x_basis[:, pos] == 1)
            return in_helper_sector & spin_up_at_pos

        def _helper_model(self, x_basis):
            if not _use_helper:
                return jnp.zeros((x_basis.shape[0],), dtype=jnp.complex128)

            helper_active = self._helper_activation_mask(x_basis)
            helper_val = jnp.full(
                (x_basis.shape[0],),
                jnp.asarray(_helper_coeff, dtype=jnp.complex128),
                dtype=jnp.complex128,
            )
            return jnp.where(
                helper_active,
                helper_val,
                jnp.zeros_like(helper_val, dtype=jnp.complex128),
            )

        def get_prerotation_helper_components(self, x_basis):
            return {
                "base_amplitude": self._eval_base_psi(x_basis).astype(jnp.complex128),
                "helper_amplitude": self._helper_model(x_basis).astype(jnp.complex128),
                "helper_active": self._helper_activation_mask(x_basis),
            }

        def get_psi_val(self, x):
            pos = self.pos_var.value
            batch_size = x.shape[0]
            idx_sigma = _batch_single_spin_config_to_index(x, pos)

            x_expanded = self._build_basis_configs(x)
            x_flat = x_expanded.reshape(-1, x.shape[1])
            psi_b = self._eval_base_psi(x_flat).reshape(batch_size, 2)

            if _use_helper:
                psi_b = psi_b + self._helper_model(x_flat).reshape(batch_size, 2)

            u_dag_lookup = self._u_dag()[idx_sigma, :]
            return jnp.sum(u_dag_lookup * psi_b, axis=1)

        def __call__(self, x):
            if _amplitude_space:
                psi_rot = self.get_psi_val(x)
                return jnp.log(psi_rot.astype(jnp.complex128) + EPS_IN_LOG)

            pos = self.pos_var.value
            batch_size = x.shape[0]
            idx_sigma = _batch_single_spin_config_to_index(x, pos)

            x_expanded = self._build_basis_configs(x)
            x_flat = x_expanded.reshape(-1, x.shape[1])
            log_psi_b = self.base(x_flat).reshape(batch_size, 2)

            u_dag_lookup = self._u_dag()[idx_sigma, :]
            return logsumexp(
                log_psi_b + jnp.log(u_dag_lookup + EPS_IN_LOG), axis=1
            )

    return Dynamic1SiteRotatedModel()


def compute_1site_prerotation_helper_base_ratio(
    model: nn.Module,
    variables,
    x_basis: jnp.ndarray,
    base_atol: float = 1e-300,
):
    """Compute helper/base amplitude diagnostics on pre-rotation basis configs."""
    if not hasattr(model, "get_prerotation_helper_components"):
        raise ValueError(
            "model must be created by create_dynamic_1site_rotated_model "
            "to expose get_prerotation_helper_components()."
        )

    x_basis = jnp.asarray(x_basis)
    components = model.apply(
        variables,
        x_basis,
        method=model.get_prerotation_helper_components,
    )

    base_amplitude = jnp.asarray(components["base_amplitude"], dtype=jnp.complex128)
    helper_amplitude = jnp.asarray(
        components["helper_amplitude"], dtype=jnp.complex128
    )
    helper_active = jnp.asarray(components["helper_active"], dtype=bool)
    abs_base = jnp.abs(base_amplitude)
    abs_helper = jnp.abs(helper_amplitude)
    base_nonzero_mask = jnp.abs(base_amplitude) > base_atol
    ratio_defined_mask = helper_active & base_nonzero_mask

    safe_base = jnp.where(
        base_nonzero_mask, base_amplitude, jnp.ones_like(base_amplitude)
    )
    helper_over_base = jnp.where(
        ratio_defined_mask,
        helper_amplitude / safe_base,
        jnp.zeros_like(helper_amplitude, dtype=jnp.complex128),
    )
    sum_abs_base = jnp.sum(abs_base)
    sum_abs_helper = jnp.sum(abs_helper)
    sum_abs_base_active = jnp.sum(abs_base[helper_active])
    sum_abs_helper_active = jnp.sum(abs_helper[helper_active])
    sum_abs_helper_over_sum_abs_base = jnp.where(
        sum_abs_base > base_atol,
        sum_abs_helper / sum_abs_base,
        jnp.asarray(jnp.nan, dtype=REAL_DTYPE),
    )
    sum_abs_helper_over_sum_abs_base_active = jnp.where(
        sum_abs_base_active > base_atol,
        sum_abs_helper_active / sum_abs_base_active,
        jnp.asarray(jnp.nan, dtype=REAL_DTYPE),
    )

    return {
        "base_amplitude": base_amplitude,
        "helper_amplitude": helper_amplitude,
        "abs_base_amplitude": abs_base,
        "abs_helper_amplitude": abs_helper,
        "helper_active": helper_active,
        "base_nonzero_mask": base_nonzero_mask,
        "ratio_defined_mask": ratio_defined_mask,
        "helper_over_base": helper_over_base,
        "sum_abs_base": sum_abs_base,
        "sum_abs_helper": sum_abs_helper,
        "sum_abs_base_active": sum_abs_base_active,
        "sum_abs_helper_active": sum_abs_helper_active,
        "sum_abs_helper_over_sum_abs_base": sum_abs_helper_over_sum_abs_base,
        "sum_abs_helper_over_sum_abs_base_active": sum_abs_helper_over_sum_abs_base_active,
    }


def create_dynamic_rotated_model(
    base_model: nn.Module,
    U: jnp.ndarray,
    helper_model_sz_constraint: Optional[float] = None,
    helper_model_coeff: float = 0.0,
) -> nn.Module:
    """Create a dynamic rotated model for general (non-Sz-conserving) U."""
    _U = U
    _helper_sz = helper_model_sz_constraint
    _helper_coeff = helper_model_coeff

    class DynamicRotatedModel(nn.Module):
        def setup(self):
            self.base = base_model
            self.i_var = self.variable('bond_info', 'i', lambda: jnp.array(0, dtype=jnp.int32))
            self.j_var = self.variable('bond_info', 'j', lambda: jnp.array(1, dtype=jnp.int32))

        def _build_basis_configs(self, x):
            i = self.i_var.value
            j = self.j_var.value
            x_expanded = jnp.tile(x[:, None, :], (1, 4, 1))
            for b in range(4):
                bit_i = (b >> 1) & 1
                bit_j = b & 1
                spin_i = jnp.where(bit_i == 0, 1, -1)
                spin_j = jnp.where(bit_j == 0, 1, -1)
                x_expanded = x_expanded.at[:, b, i].set(spin_i)
                x_expanded = x_expanded.at[:, b, j].set(spin_j)
            return x_expanded

        def get_psi_val(self, x):
            i = self.i_var.value
            j = self.j_var.value
            batch_size = x.shape[0]
            idx_sigma = batch_two_spin_config_to_index(x, i, j)

            x_expanded = self._build_basis_configs(x)
            x_flat = x_expanded.reshape(-1, x.shape[1])

            if hasattr(self.base, 'get_psi_val'):
                psi_b = self.base.get_psi_val(x_flat).reshape(batch_size, 4)
            else:
                psi_b = jnp.exp(self.base(x_flat).reshape(batch_size, 4))

            if _helper_sz is not None:
                twice_sz_b = jnp.sum(x_flat, axis=-1).reshape(batch_size, 4)
                in_helper_sector = (twice_sz_b == int(2 * _helper_sz))
                psi_b = psi_b + jnp.where(in_helper_sector, _helper_coeff, 0.0)

            U_dag = jnp.asarray(_U).conj().T
            U_dag_lookup = U_dag[idx_sigma, :]
            return jnp.sum(U_dag_lookup * psi_b, axis=1)

        def __call__(self, x):
            if _helper_sz is not None:
                psi_rot = self.get_psi_val(x)
                return jnp.log(psi_rot.astype(jnp.complex128) + EPS_IN_LOG)
            else:
                i = self.i_var.value
                j = self.j_var.value
                batch_size = x.shape[0]
                idx_sigma = batch_two_spin_config_to_index(x, i, j)

                x_expanded = self._build_basis_configs(x)
                x_flat = x_expanded.reshape(-1, x.shape[1])
                log_psi_b = self.base(x_flat).reshape(batch_size, 4)

                U_dag = jnp.asarray(_U).conj().T
                U_dag_lookup = U_dag[idx_sigma, :]
                log_psi_rot = logsumexp(
                    log_psi_b + jnp.log(U_dag_lookup + EPS_IN_LOG), axis=1
                )
                return log_psi_rot

    return DynamicRotatedModel()
