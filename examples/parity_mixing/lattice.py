"""Lattice graph shared by the parity-mixing demo scripts (square lattice or chain)."""
from __future__ import annotations

import netket as nk


def make_graph(lx: int, ly: int, pbc: bool) -> nk.graph.Graph:
    """``Grid([lx, ly])`` for a 2D lattice, ``Chain(lx)`` when ly == 1."""
    if ly == 1:
        return nk.graph.Chain(lx, pbc=pbc)
    return nk.graph.Grid([lx, ly], pbc=pbc)
