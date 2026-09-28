"""Get Stock Qiskit's initial placement without compiling its final circuit."""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.transpiler import StagedPassManager

import hybrid_level3 as stock


def stock_initial_layout(qc: QuantumCircuit, backend, *, seed_transpiler: int) -> tuple[int, ...]:
    """Run Stock Level-3 through routing and return its starting placement.

    The routing stage can revise the layout stage's first choice. This shorter
    prefix matches full Stock in most tested cases, though not every case.
    """
    manager = stock._build_level3(
        backend.target, initial_layout=None, seed_transpiler=seed_transpiler
    )
    stages = ("init", "layout", "routing")
    prefix = StagedPassManager(
        stages=stages, **{stage: getattr(manager, stage) for stage in stages}
    )
    placed = prefix.run(qc.copy())
    if placed.layout is None:
        raise ValueError("Qiskit did not return an initial layout")
    positions = tuple(
        int(site) for site in placed.layout.initial_index_layout()[: qc.num_qubits]
    )
    if len(positions) != qc.num_qubits or len(set(positions)) != qc.num_qubits:
        raise ValueError("Qiskit returned an incomplete initial layout")
    if any(site < 0 or site >= backend.num_qubits for site in positions):
        raise ValueError("Qiskit returned a site outside the target")
    return positions


def physical_line(cmap, length: int) -> tuple[int, ...]:
    """Find a deterministic connected line on the target."""
    adjacent = [set() for _ in range(cmap.size())]
    for a, b in cmap.get_edges():
        adjacent[a].add(b)
        adjacent[b].add(a)
    explored = 0

    def extend(path: list[int], used: set[int]) -> tuple[int, ...] | None:
        nonlocal explored
        explored += 1
        if explored > 2_000_000:
            raise RuntimeError("Could not find a physical line within the search limit")
        if len(path) == length:
            return tuple(path)
        for site in sorted(adjacent[path[-1]] - used,
                           key=lambda q: (-len(adjacent[q] - used), q)):
            result = extend(path + [site], used | {site})
            if result is not None:
                return result
        return None

    for start in sorted(range(cmap.size()), key=lambda q: (-len(adjacent[q]), q)):
        result = extend([start], {start})
        if result is not None:
            return result
    raise ValueError(f"Target has no {length}-qubit simple path")
