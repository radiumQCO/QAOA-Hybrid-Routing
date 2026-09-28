"""Fast large-QAOA line routing built on the frozen v2.0.0 router.

The 16–24q behavior is unchanged. Supported 25–64q circuits compile only the
chosen line candidate after a cheap Rust search.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from qiskit import QuantumCircuit

import hybrid_level3 as base
from hybrid_level3_v2 import Candidate, _compile_candidate, compile_hybrid_dense_v2
from layout_seed import physical_line
from rust_router_core import optimise_line_layout_rust, route_line_layers_rust


def compile_hybrid_line_next(
    qc: QuantumCircuit,
    backend,
    *,
    seed_transpiler: int = 11,
    route_verifier=None,
    return_candidate: bool = False,
    seed11_positions: tuple[int, ...] | None = None,
):
    """Compile one optimized line; on medium graphs rank three Rust starts first."""
    shape = base.inspect_qaoa_shape(qc)
    n = qc.num_qubits
    medium = 0.45 <= shape.density < 0.65
    dense = shape.density >= 0.65
    if not (24 < n <= 64 and backend.num_qubits >= n and shape.supported
            and (medium or dense)):
        return compile_hybrid_dense_v2(
            qc, backend, seed_transpiler=seed_transpiler,
            route_verifier=route_verifier, return_candidate=return_candidate)

    started = perf_counter()
    cmap = backend.coupling_map
    path = physical_line(cmap, n)
    seeds = (seed_transpiler, seed_transpiler + 2, seed_transpiler + 6) if medium else (seed_transpiler,)
    routes = []
    seen = set()
    for seed in seeds:
        positions = (tuple(seed11_positions) if seed == seed_transpiler and seed11_positions is not None
                     else optimise_line_layout_rust(qc, seed=seed, max_seconds=0.15))
        if set(positions) != set(range(n)):
            raise ValueError("seed11_positions must permute the logical line sites")
        if positions in seen:
            continue
        seen.add(positions)
        layout = tuple(path[site] for site in positions)
        route_started = perf_counter()
        routed = route_line_layers_rust(qc.copy(), cmap, path, layout)
        candidate = Candidate(layout, routed, perf_counter() - route_started)
        routes.append((seed, candidate))
    seed, chosen = min(routes, key=lambda item: (item[1].routed.swaps, item[0]))
    if route_verifier is not None:
        route_verifier("line_search", qc, chosen.routed, chosen.layout, cmap)
    _compile_candidate(chosen, backend, seed_transpiler, shape.density)
    family = "medium" if medium else "dense"
    result = replace(
        chosen.compiled,
        route_mode=f"v2_rust_{family}_line_seed{seed}",
        selector_reason=f"{n}q line: {len(routes)} searched orders; "
                        f"fewest symbolic SWAPs, selected seed {seed}",
        router_seconds=max(0.0, perf_counter() - started - chosen.compiled.level3_seconds),
    )
    return (result, chosen) if return_candidate else result
