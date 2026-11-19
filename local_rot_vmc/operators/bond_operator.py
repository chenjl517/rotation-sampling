"""Custom Hamiltonian operators for NetKet."""

import numpy as np
from netket.hilbert import Spin


def create_2site_rotated_bond_operator(
    N: int,
    i: int,
    j: int,
    model: str,
    J: float,
    g: float,
    U: np.ndarray,
    hilbert=None,
    dtype=np.complex128,
    delta: float = 1.0
):
    """Create a two-site rotated bond operator H_rot = U† H_bond U on sites (i, j)."""
    if hilbert is None:
        hilbert = Spin(s=0.5, N=N)

    if U.shape != (4, 4):
        raise ValueError(f"U must be 4×4, got shape {U.shape}")

    H_4x4 = _construct_bond_hamiltonian_4x4(model, J, g, coordination_number=4, delta=delta)

    H_rot_4x4 = U.conj().T @ H_4x4 @ U

    H_rot = _decompose_4x4_to_local_operator(H_rot_4x4, i, j, hilbert, dtype)

    return H_rot


def _convert_1site_unitary_to_up_down_basis(U_2x2: np.ndarray, dtype=np.complex128) -> np.ndarray:
    if U_2x2.shape != (2, 2):
        raise ValueError(f"U must be 2x2, got shape {U_2x2.shape}")

    swap = np.array([[0, 1], [1, 0]], dtype=dtype)
    U_2x2 = np.asarray(U_2x2, dtype=dtype)
    return swap @ U_2x2 @ swap


def create_1site_rotated_bond_operator(
    N: int,
    i: int,
    j: int,
    active_site: int,
    model: str,
    J: float,
    g: float,
    U: np.ndarray,
    hilbert=None,
    dtype=np.complex128,
    delta: float = 1.0,
):
    """Create a bond operator with single-site unitary rotation."""
    if active_site not in (i, j):
        raise ValueError(
            f"active_site={active_site} must belong to bond ({i}, {j})"
        )

    if hilbert is None:
        hilbert = Spin(s=0.5, N=N)

    U_2x2 = np.asarray(U)
    if U_2x2.shape != (2, 2):
        raise ValueError(f"U must be 2x2, got shape {U_2x2.shape}")

    H_4x4 = _construct_bond_hamiltonian_4x4(
        model, J, g, coordination_number=4, delta=delta
    )
    U_up_down = _convert_1site_unitary_to_up_down_basis(U_2x2, dtype=dtype)
    eye = np.eye(2, dtype=dtype)

    if active_site == i:
        U_full = np.kron(U_up_down, eye)
    else:
        U_full = np.kron(eye, U_up_down)

    H_rot_4x4 = U_full.conj().T @ H_4x4 @ U_full
    return _decompose_4x4_to_local_operator(H_rot_4x4, i, j, hilbert, dtype)


def _construct_bond_hamiltonian_4x4(model: str, J: float, g: float, coordination_number=4, delta: float = 1.0) -> np.ndarray:
    H_4x4 = np.zeros((4, 4), dtype=np.complex128)

    if model == "TFIM":
        H_4x4 += -J * np.diag([1, -1, -1, 1])

        xI = -(g/coordination_number) * np.array([[0, 0, 1, 0],
                                [0, 0, 0, 1],
                                [1, 0, 0, 0],
                                [0, 1, 0, 0]], dtype=np.complex128)
        Ix = -(g/coordination_number) * np.array([[0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0]], dtype=np.complex128)

        H_4x4 += xI + Ix

    elif model == "Heisenberg":
        coeff = J / 4

        xx = coeff * np.array([[0, 0, 0, 1],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [1, 0, 0, 0]], dtype=np.complex128)

        yy = coeff * np.array([[0, 0, 0, -1],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [-1, 0, 0, 0]], dtype=np.complex128)

        zz = coeff * np.diag([1, -1, -1, 1])

        H_4x4 = xx + yy + zz

    elif model == "XXZ":
        coeff = J / 4

        xx = coeff * np.array([[0, 0, 0, 1],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [1, 0, 0, 0]], dtype=np.complex128)

        yy = coeff * np.array([[0, 0, 0, -1],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [-1, 0, 0, 0]], dtype=np.complex128)

        zz = (delta/4)* np.diag([1, -1, -1, 1])

        H_4x4 = xx + yy + zz

    else:
        raise ValueError(f"Unknown model: {model}")

    return H_4x4


def create_3site_rotated_operator(
    N: int,
    i: int,
    j: int,
    exci_bond: tuple,
    model: str,
    J: float,
    g: float,
    U: np.ndarray,
    hilbert=None,
    dtype=np.complex128,
    delta: float = 1.0,
):
    """Create U†_{exci_bond} h_ij U_{exci_bond} for a bond (i, j) sharing one site with ``exci_bond``."""
    if hilbert is None:
        hilbert = Spin(s=0.5, N=N)

    U_4x4 = np.asarray(U)
    if U_4x4.shape != (4, 4):
        raise ValueError(f"U must be 4×4, got shape {U_4x4.shape}")

    exci_sites = tuple(exci_bond)
    bond_sites = (i, j)
    i_e, j_e = exci_sites

    shared = set(exci_sites) & set(bond_sites)
    if len(shared) != 1:
        raise ValueError(
            f"bond ({i}, {j}) must share exactly one site with "
            f"exci_bond={exci_sites}, got shared={shared}"
        )
    s = shared.pop()
    k = (set(bond_sites) - {s}).pop()

    h_4x4 = _construct_bond_hamiltonian_4x4(model, J, g, coordination_number=4, delta=delta)

    U_full = np.kron(U_4x4, np.eye(2, dtype=dtype))

    if s == i_e:
        h_t = h_4x4.reshape(2, 2, 2, 2)
        h_full = np.einsum('acAC,bB->abcABC', h_t, np.eye(2, dtype=dtype)).reshape(8, 8)
    else:
        h_full = np.kron(np.eye(2, dtype=dtype), h_4x4)

    M_8x8 = U_full.conj().T @ h_full @ U_full

    return _decompose_8x8_to_local_operator(M_8x8, i_e, j_e, k, hilbert, dtype)


def _decompose_8x8_to_local_operator(
    M_8x8: np.ndarray,
    i: int,
    j: int,
    k: int,
    hilbert,
    dtype,
):
    from netket.operator import LocalOperator as LO

    pauli_set = [
        np.eye(2, dtype=dtype),
        np.array([[0, 1], [1, 0]], dtype=dtype),
        np.array([[0, -1j], [1j, 0]], dtype=dtype),
        np.array([[1, 0], [0, -1]], dtype=dtype),
    ]

    H_new = LO(hilbert, dtype=dtype)

    for ai in range(4):
        for bi in range(4):
            for ci in range(4):
                basis_op = np.kron(np.kron(pauli_set[ai], pauli_set[bi]), pauli_set[ci])
                coeff = np.trace(basis_op.conj().T @ M_8x8) / 8

                if np.abs(coeff) < 1e-14:
                    continue

                active_sites = []
                active_mats = []
                if ai != 0:
                    active_sites.append(i)
                    active_mats.append(pauli_set[ai])
                if bi != 0:
                    active_sites.append(j)
                    active_mats.append(pauli_set[bi])
                if ci != 0:
                    active_sites.append(k)
                    active_mats.append(pauli_set[ci])

                if not active_sites:
                    H_new = H_new + LO(hilbert, np.eye(2, dtype=dtype), [i], dtype=dtype) * coeff
                elif len(active_sites) == 1:
                    H_new = H_new + LO(hilbert, active_mats[0], active_sites, dtype=dtype) * coeff
                elif len(active_sites) == 2:
                    mat_2 = np.kron(active_mats[0], active_mats[1])
                    H_new = H_new + LO(hilbert, mat_2, active_sites, dtype=dtype) * coeff
                else:
                    mat_3 = np.kron(np.kron(active_mats[0], active_mats[1]), active_mats[2])
                    H_new = H_new + LO(hilbert, mat_3, active_sites, dtype=dtype) * coeff

    return H_new


def _decompose_4x4_to_local_operator(
    H_4x4: np.ndarray,
    i: int,
    j: int,
    hilbert,
    dtype
):
    from netket.operator import LocalOperator as LO

    pauli_set = [
        np.eye(2, dtype=dtype),
        np.array([[0, 1], [1, 0]], dtype=dtype),
        np.array([[0, -1j], [1j, 0]], dtype=dtype),
        np.array([[1, 0], [0, -1]], dtype=dtype)
    ]

    H_new = LO(hilbert, dtype=dtype)

    for a in range(4):
        for b in range(4):
            pauli_prod = np.kron(pauli_set[a], pauli_set[b])
            coeff = np.trace(pauli_prod.conj().T @ H_4x4) / 4

            if np.abs(coeff) < 1e-14:
                continue

            if a == 0 and b != 0:
                H_new = H_new + LO(hilbert, pauli_set[b], [j], dtype=dtype) * coeff
            elif b == 0 and a != 0:
                H_new = H_new + LO(hilbert, pauli_set[a], [i], dtype=dtype) * coeff
            else:
                mat_2site = np.kron(pauli_set[a], pauli_set[b])
                H_new = H_new + LO(hilbert, mat_2site, [i, j], dtype=dtype) * coeff

    return H_new
