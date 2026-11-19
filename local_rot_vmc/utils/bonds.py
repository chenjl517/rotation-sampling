"""Bond generation and grouping utilities."""

from collections.abc import Iterable
from typing import Any, List, Tuple


def get_2d_nnn_bonds(L: int, pbc: bool = True) -> List[Tuple[int, int]]:
    """Generate next-nearest-neighbor (diagonal) bonds for an L x L square lattice."""
    bonds_set: set = set()
    for y in range(L):
        for x in range(L):
            site = x + L * y
            nx1, ny1 = (x + 1) % L, (y + 1) % L
            if pbc or (x + 1 < L and y + 1 < L):
                neighbor1 = nx1 + L * ny1
                bond = (min(site, neighbor1), max(site, neighbor1))
                bonds_set.add(bond)
            nx2, ny2 = (x + 1) % L, (y - 1) % L
            if pbc or (x + 1 < L and y - 1 >= 0):
                neighbor2 = nx2 + L * ny2
                bond = (min(site, neighbor2), max(site, neighbor2))
                bonds_set.add(bond)
    return sorted(bonds_set)


def extract_bonds_from_graph(graph: Any) -> List[Tuple[int, int]]:
    """Extract undirected bond list from a NetKet graph/lattice object."""
    if not hasattr(graph, "edges"):
        raise ValueError("graph must provide an edges() method.")

    raw_edges: Iterable[Any] = graph.edges()
    bonds_set: set[Tuple[int, int]] = set()
    for edge in raw_edges:
        if len(edge) < 2:
            raise ValueError(f"Invalid edge format: {edge!r}")
        i = int(edge[0])
        j = int(edge[1])
        if i == j:
            continue
        if i > j:
            i, j = j, i
        bonds_set.add((i, j))

    return sorted(bonds_set)


def _canonical_bonds(bonds: List[Tuple[int, int]]) -> set[Tuple[int, int]]:
    return {
        (min(int(i), int(j)), max(int(i), int(j)))
        for i, j in bonds
        if int(i) != int(j)
    }


def _graph_geometry(graph: Any) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    if not hasattr(graph, "extent"):
        raise ValueError("graph must expose an 'extent' attribute to build edge groups explicitly.")

    extent = tuple(int(x) for x in tuple(graph.extent))
    if not extent:
        raise ValueError("graph.extent must be non-empty.")

    raw_pbc = getattr(graph, "pbc", None)
    if raw_pbc is None:
        pbc = (False,) * len(extent)
    elif isinstance(raw_pbc, (bool, int)):
        pbc = (bool(raw_pbc),) * len(extent)
    else:
        pbc = tuple(bool(x) for x in tuple(raw_pbc))
        if len(pbc) == 1 and len(extent) > 1:
            pbc = pbc * len(extent)

    if len(pbc) != len(extent):
        raise ValueError(f"graph.pbc dimensionality mismatch: extent={extent}, pbc={pbc}")

    return extent, pbc


def _build_1d_edge_groups_from_graph(
    bonds: List[Tuple[int, int]],
    n_nodes: int,
    *,
    pbc: bool,
) -> List[List[Tuple[int, int]]]:
    canon = _canonical_bonds(bonds)
    if n_nodes <= 1:
        return []

    allowed = {(i, i + 1) for i in range(n_nodes - 1)}
    if pbc and n_nodes > 2:
        allowed.add((0, n_nodes - 1))

    if canon != allowed:
        raise ValueError(
            f"1D graph bonds do not match nearest-neighbor {'PBC' if pbc else 'OBC'} chain edges: "
            f"expected {len(allowed)} bonds, got {len(canon)}."
        )

    n_groups = 2 if (not pbc or n_nodes % 2 == 0) else 3
    groups: List[List[Tuple[int, int]]] = [[] for _ in range(n_groups)]

    for i in range(n_nodes - 1):
        groups[i % n_groups].append((i, i + 1))
    if pbc and n_nodes > 2:
        groups[(n_nodes - 1) % n_groups].append((0, n_nodes - 1))

    return groups


def _build_2d_edge_groups_from_graph(
    bonds: List[Tuple[int, int]],
    extent: tuple[int, int],
    *,
    pbc_x: bool,
    pbc_y: bool,
) -> List[List[Tuple[int, int]]]:
    lx, ly = map(int, extent)
    n_nodes = lx * ly
    canon = _canonical_bonds(bonds)

    def site(x: int, y: int) -> int:
        return int(x * ly + y)

    allowed: set[Tuple[int, int]] = set()
    for y in range(ly):
        for x in range(lx):
            if x + 1 < lx or pbc_x:
                allowed.add(tuple(sorted((site(x, y), site((x + 1) % lx, y)))))
            if y + 1 < ly or pbc_y:
                allowed.add(tuple(sorted((site(x, y), site(x, (y + 1) % ly)))))

    if canon != allowed:
        raise ValueError(
            f"2D graph bonds do not match nearest-neighbor square-lattice edges for extent={extent}, "
            f"pbc=({pbc_x}, {pbc_y})."
        )

    n_h_groups = 2 if (not pbc_x or lx % 2 == 0) else 3
    n_v_groups = 2 if (not pbc_y or ly % 2 == 0) else 3
    h_groups: List[List[Tuple[int, int]]] = [[] for _ in range(n_h_groups)]
    v_groups: List[List[Tuple[int, int]]] = [[] for _ in range(n_v_groups)]

    for y in range(ly):
        for x in range(lx):
            if x + 1 < lx or pbc_x:
                edge = tuple(sorted((site(x, y), site((x + 1) % lx, y))))
                h_groups[x % n_h_groups].append(edge)
            if y + 1 < ly or pbc_y:
                edge = tuple(sorted((site(x, y), site(x, (y + 1) % ly))))
                v_groups[y % n_v_groups].append(edge)

    return h_groups + v_groups


def build_edge_groups(graph: Any, bonds: List[Tuple[int, int]] | None = None) -> List[List[Tuple[int, int]]]:
    """Build deterministic non-overlapping bond groups from explicit graph geometry."""
    if not hasattr(graph, "n_nodes"):
        raise ValueError("graph must expose an 'n_nodes' attribute.")

    n_nodes = int(graph.n_nodes)
    if n_nodes <= 0:
        raise ValueError(f"graph.n_nodes must be positive, got {n_nodes}")

    if bonds is None:
        bonds = extract_bonds_from_graph(graph)

    extent, pbc = _graph_geometry(graph)
    if len(extent) == 1:
        return _build_1d_edge_groups_from_graph(bonds, n_nodes, pbc=bool(pbc[0]))
    if len(extent) == 2:
        return _build_2d_edge_groups_from_graph(
            bonds,
            (int(extent[0]), int(extent[1])),
            pbc_x=bool(pbc[0]),
            pbc_y=bool(pbc[1]),
        )

    raise ValueError(f"Unsupported graph dimensionality for build_edge_groups: extent={extent}")


def reorder_bond_groups(
    bond_groups: List[List[Tuple[int, int]]],
    exci_bond: Tuple[int, int],
) -> List[List[Tuple[int, int]]]:
    """Reorder bond groups cyclically so the group containing *exci_bond* comes first."""
    import copy

    canon = (min(exci_bond), max(exci_bond))

    target_idx = None
    for gidx, group in enumerate(bond_groups):
        if canon in group:
            target_idx = gidx
            break

    if target_idx is None:
        raise ValueError(
            f"Excitation bond {exci_bond} not found in any bond group."
        )

    reordered = copy.deepcopy(bond_groups[target_idx:] + bond_groups[:target_idx])

    first_group = reordered[0]
    first_idx = first_group.index(canon)
    reordered[0] = first_group[first_idx:] + first_group[:first_idx]

    return reordered
