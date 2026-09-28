"""Hybrid v2.0.0 layout search and complete-circuit selection."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from time import perf_counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate

import routing_metrics as metrics
import hybrid_level3 as v1
from layout_seed import physical_line, stock_initial_layout
from rust_router_core import (
    compile_rust_core_level3, optimise_line_layout_rust,
    route_beam_layers_rust, route_line_layers_rust,
)


# Keep the 16–24q settings from the saved baseline.
MIN_ROUTED_QUBITS = 16
MAX_ROUTED_QUBITS = 64
FROZEN_LINE_QUBITS = 24


@dataclass
class Candidate:
    layout: tuple[int, ...]
    routed: v1.RoutedCircuit
    router_seconds: float
    compiled: v1.CompileResult | None = None
    native_2q: int | None = None
    twoq_depth: int | None = None
    error_cost: float | None = None


class LayoutSearchFailed(RuntimeError):
    """No tested layout produced a complete Rust route."""


def _distances(cmap, n: int) -> list[list[int]]:
    adj = [set() for _ in range(n)]
    for a, b in cmap.get_edges():
        adj[a].add(b)
        adj[b].add(a)
    out = []
    for start in range(n):
        distances = [n + 1] * n
        distances[start] = 0
        queue = [start]
        for site in queue:
            for neighbor in sorted(adj[site]):
                if distances[neighbor] > distances[site] + 1:
                    distances[neighbor] = distances[site] + 1
                    queue.append(neighbor)
        if max(distances) > n:
            raise ValueError("The coupling map must be connected")
        out.append(distances)
    return out


def _full_target_layouts(qc: QuantumCircuit, cmap, pairs, count: int,
                         preferred_layout: tuple[int, ...] | None) -> list[tuple[int, ...]]:
    """Spread connected starts over a larger physical target."""
    logical_count = qc.num_qubits
    physical_count = cmap.size()
    if preferred_layout is not None:
        if (len(preferred_layout) != logical_count or len(set(preferred_layout)) != logical_count
                or any(q < 0 or q >= physical_count for q in preferred_layout)):
            raise ValueError("The preferred layout is not a valid physical placement")
    layouts = [preferred_layout if preferred_layout is not None else tuple(range(logical_count))]
    if count == 1:
        return layouts
    distances = _distances(cmap, physical_count)
    neighbors = [set() for _ in range(physical_count)]
    for a, b in cmap.get_edges():
        neighbors[a].add(b)
        neighbors[b].add(a)
    logical_neighbors = [set() for _ in range(logical_count)]
    for u, v in pairs:
        logical_neighbors[u].add(v)
        logical_neighbors[v].add(u)
    logical_order = sorted(range(logical_count), key=lambda q: (-len(logical_neighbors[q]), q))

    seen = set(layouts)
    scored = []
    for seed in range(physical_count):
        region = {seed}
        while len(region) < logical_count:
            frontier = set().union(*(neighbors[q] for q in region)) - region
            if not frontier:
                raise ValueError(f"The physical target has no connected {logical_count}-qubit region")
            region.add(min(frontier, key=lambda q: (-len(neighbors[q] & region),
                                                    -len(neighbors[q]), q)))
        assigned = {logical_order[0]: seed}
        free = region - {seed}
        for logical in logical_order[1:]:
            interacting = logical_neighbors[logical] & assigned.keys()
            site = min(free, key=lambda q: (
                sum(distances[q][assigned[other]] for other in interacting),
                sum(distances[q][other] for other in assigned.values()),
                -len(neighbors[q]), q,
            ))
            assigned[logical] = site
            free.remove(site)
        layout = tuple(assigned[q] for q in range(logical_count))
        if layout not in seen:
            distance_cost = sum(distances[layout[u]][layout[v]] for u, v in pairs)
            scored.append((distance_cost, seed, layout))
            seen.add(layout)
    scored.sort()
    layouts.extend(layout for _, _, layout in scored[:count - len(layouts)])
    if len(layouts) != count:
        raise ValueError(f"Could not generate {count} distinct starting layouts")
    return layouts


def starting_layouts(qc: QuantumCircuit, cmap, count: int = 8,
                     preferred_layout: tuple[int, ...] | None = None) -> list[tuple[int, ...]]:
    """Identity plus degree-guided placements, deterministic for a given circuit."""
    n = qc.num_qubits
    if cmap.size() < n or count < 1:
        raise ValueError("Expected at least as many physical qubits as logical qubits")
    layer = next((inst for inst in qc.data if isinstance(inst.operation, PauliEvolutionGate)), None)
    if layer is None:
        raise ValueError("No ZZ layer for layout search")
    pairs = v1._layer_terms(qc, layer)
    if cmap.size() > n:
        return _full_target_layouts(qc, cmap, pairs, count, preferred_layout)
    logical_neighbors = [set() for _ in range(n)]
    physical_neighbors = [set() for _ in range(n)]
    for u, v in pairs:
        logical_neighbors[u].add(v)
        logical_neighbors[v].add(u)
    for a, b in cmap.get_edges():
        physical_neighbors[a].add(b)
        physical_neighbors[b].add(a)
    distances = _distances(cmap, n)
    physical_order = sorted(range(n), key=lambda q: (-len(physical_neighbors[q]), q))
    logical_order = sorted(range(n), key=lambda q: (-len(logical_neighbors[q]), q))
    layouts = [tuple(range(n))]
    seen = set(layouts)
    for start in physical_order:
        assigned = {logical_order[0]: start}
        free = set(range(n)) - {start}
        for logical in logical_order[1:]:
            neighbors = logical_neighbors[logical] & assigned.keys()
            chosen = min(
                free,
                key=lambda site: (
                    sum(distances[site][assigned[other]] for other in neighbors),
                    sum(distances[site][other] for other in assigned.values()),
                    -len(physical_neighbors[site]),
                    site,
                ),
            )
            assigned[logical] = chosen
            free.remove(chosen)
        layout = tuple(assigned[q] for q in range(n))
        if layout not in seen:
            layouts.append(layout)
            seen.add(layout)
        if len(layouts) == count:
            break
    rng = random.Random(sum((u + 1) * 257 + v for u, v in pairs) + n * 65537)
    while len(layouts) < count:
        layout = tuple(rng.sample(range(n), n))
        if layout not in seen:
            layouts.append(layout)
            seen.add(layout)
    return layouts


def _error_cost(circuit: QuantumCircuit, target) -> float | None:
    """Approximate 2Q error exposure; unavailable when any calibration is missing."""
    total = 0.0
    for inst in circuit.data:
        if inst.operation.num_qubits != 2:
            continue
        sites = tuple(circuit.find_bit(q).index for q in inst.qubits)
        try:
            properties = target[inst.operation.name][sites]
            error = properties.error if properties is not None else None
        except (KeyError, TypeError):
            return None
        if error is None or not math.isfinite(error) or not 0 <= error < 1:
            return None
        total -= math.log1p(-error)
    return total


def _compile_candidate(candidate: Candidate, backend, seed_transpiler: int, density: float) -> None:
    start = perf_counter()
    # Rust already placed the SWAPs, so Qiskit must not reroute this circuit.
    manager = v1._build_level3(
        backend.target,
        initial_layout=list(range(backend.num_qubits)),
        seed_transpiler=seed_transpiler,
        routing_method="none",
    )
    circuit = manager.run(candidate.routed.circuit)
    level3_seconds = perf_counter() - start
    physical_output = v1._final_positions(circuit, backend.num_qubits)
    positions = [physical_output[q] for q in candidate.routed.final_positions]
    candidate.compiled = v1.CompileResult(
        circuit=circuit,
        final_positions=positions,
        route_mode="v2_rust_layout_search",
        selector_reason="degree-guided multi-start layout search",
        density=density,
        router_seconds=candidate.router_seconds,
        level3_seconds=level3_seconds,
        swaps_before_level3=candidate.routed.swaps,
    )
    candidate.native_2q = sum(inst.operation.num_qubits == 2 for inst in circuit.data)
    candidate.twoq_depth = metrics.twoq_depth(circuit)
    candidate.error_cost = _error_cost(circuit, backend.target)


def compare_variants(
    qc: QuantumCircuit,
    backend,
    *,
    seed_transpiler: int,
    layouts: int = 8,
    finalists: int = 4,
    beam_width: int = v1.DEFAULT_BEAM_WIDTH,
    beam_seconds: float = v1.DEFAULT_BEAM_SECONDS,
    preferred_layout: tuple[int, ...] | None = None,
) -> tuple[dict[str, v1.CompileResult | None], list[Candidate]]:
    """Return multi-start, native-2Q tournament, and error-score variants."""
    shape = v1.inspect_qaoa_shape(qc)
    if (not MIN_ROUTED_QUBITS <= qc.num_qubits <= MAX_ROUTED_QUBITS
            or backend.num_qubits < qc.num_qubits or not shape.supported
            or not v1.MIN_BEAM_DENSITY <= shape.density <= v1.MAX_BEAM_DENSITY):
        raise ValueError(f"v2.0.0 research needs a supported 16–{MAX_ROUTED_QUBITS} logical-qubit QAOA case")
    if finalists < 1:
        raise ValueError("finalists must be positive")

    candidates = []
    for layout in starting_layouts(qc, backend.coupling_map, layouts, preferred_layout):
        start = perf_counter()
        try:
            routed = route_beam_layers_rust(
                qc.copy(), backend.coupling_map, list(layout),
                beam_width=beam_width, max_seconds=beam_seconds,
            )
        except (v1.BeamBudgetExceeded, v1.BeamRoutingFailed):
            continue
        candidates.append(Candidate(layout, routed, perf_counter() - start))
    if not candidates:
        raise LayoutSearchFailed("All v2.0.0 layout searches failed")

    # Routing SWAP count is cheap; preserve the available control layout for compilation.
    ranked = sorted(candidates, key=lambda c: (c.routed.swaps, c.routed.circuit.depth(), c.layout))
    first = ranked[0]
    identity = next((c for c in candidates if c.layout == tuple(range(qc.num_qubits))), None)
    preferred = next((c for c in candidates if c.layout == preferred_layout), None)
    chosen = [first]
    for control in (preferred, identity):
        if (control is not None and all(control is not existing for existing in chosen)
                and len(chosen) < finalists):
            chosen.append(control)
    chosen.extend(c for c in ranked if all(c is not existing for existing in chosen))
    chosen = chosen[:finalists]
    for candidate in chosen:
        _compile_candidate(candidate, backend, seed_transpiler, shape.density)

    by_cz = min(chosen, key=lambda c: (c.native_2q, c.twoq_depth, c.compiled.circuit.depth()))
    calibrated = [c for c in chosen if c.error_cost is not None]
    by_error = (min(calibrated, key=lambda c: (c.error_cost, c.native_2q)).compiled
                if len(calibrated) == len(chosen) else None)
    return {
        "multistart": first.compiled,
        "tournament": by_cz.compiled,
        "error_aware": by_error,
    }, candidates


def compile_hybrid_level3_v2(
    qc: QuantumCircuit,
    backend,
    *,
    seed_transpiler: int,
    layouts: int | None = None,
    finalists: int = 4,
    beam_seconds: float | None = None,
    quality_guard: bool = True,
) -> v1.CompileResult:
    """Compile a routed finalist, optionally checking it against full Stock."""
    start = perf_counter()
    shape = v1.inspect_qaoa_shape(qc)
    activated = (MIN_ROUTED_QUBITS <= qc.num_qubits <= MAX_ROUTED_QUBITS
                 and backend.num_qubits >= qc.num_qubits and shape.supported
                 and v1.MIN_BEAM_DENSITY <= shape.density <= v1.MAX_BEAM_DENSITY)
    if not activated:
        return compile_rust_core_level3(qc, backend, seed_transpiler=seed_transpiler)
    layout_count = layouts if layouts is not None else (32 if backend.num_qubits == 16 else 1)
    search_seconds = beam_seconds if beam_seconds is not None else (
        v1.DEFAULT_BEAM_SECONDS if backend.num_qubits == 16 else 1.0)
    if layout_count < 1 or search_seconds <= 0:
        raise ValueError("layouts and beam_seconds must be positive")
    # Raw mode needs Stock's placement, but not its final circuit or quality vote.
    stock = (v1.compile_stock_level3(qc.copy(), backend, seed_transpiler=seed_transpiler)
             if quality_guard else None)
    preferred_layout = None
    if backend.num_qubits > qc.num_qubits:
        preferred_layout = (tuple(stock.circuit.layout.initial_index_layout()[:qc.num_qubits])
                            if stock is not None else
                            stock_initial_layout(qc, backend, seed_transpiler=seed_transpiler))
    undirected_edges = {tuple(sorted((a, b))) for a, b in backend.coupling_map.get_edges()}
    search_failed = False
    if (qc.num_qubits == 16 and backend.num_qubits == 16
            and len(undirected_edges) == backend.num_qubits - 1):
        # The fixed IBM tree-region test found no 2Q gain from v2.0.0 and large search costs.
        hybrid = compile_rust_core_level3(qc, backend, seed_transpiler=seed_transpiler)
    else:
        try:
            variants, _ = compare_variants(
                qc, backend, seed_transpiler=seed_transpiler,
                layouts=layout_count, finalists=finalists, beam_seconds=search_seconds,
                preferred_layout=preferred_layout,
            )
            hybrid = variants["tournament"]
        except LayoutSearchFailed:
            search_failed = True
            if backend.num_qubits > qc.num_qubits or qc.num_qubits > 16:
                if stock is None:
                    stock = v1.compile_stock_level3(qc.copy(), backend,
                                                    seed_transpiler=seed_transpiler)
                hybrid = stock
            else:
                hybrid = compile_rust_core_level3(qc, backend,
                                                  seed_transpiler=seed_transpiler)
    hybrid_key = (sum(inst.operation.num_qubits == 2 for inst in hybrid.circuit.data),
                  metrics.twoq_depth(hybrid.circuit))
    stock_key = ((sum(inst.operation.num_qubits == 2 for inst in stock.circuit.data),
                  metrics.twoq_depth(stock.circuit)) if stock is not None else None)
    selected = stock if stock_key is not None and stock_key < hybrid_key else hybrid
    elapsed = perf_counter() - start
    return replace(
        selected,
        route_mode=("v2_stock_search_fallback" if search_failed and selected is stock
                    else "v2_stock_quality_guard" if selected is stock and selected is not hybrid
                    else "v2_rust_layout_tournament" if selected.route_mode == "v2_rust_layout_search"
                    else selected.route_mode),
        selector_reason=(f"{layout_count} starts, {search_seconds:g}s search limit; "
                         f"Hybrid {hybrid_key}, Stock {stock_key if stock_key is not None else 'not run'}"),
        # This includes all layout attempts, finalist compilations, and Stock comparison.
        router_seconds=max(0.0, elapsed - selected.level3_seconds),
    )


def compile_hybrid_dense_v2(
    qc: QuantumCircuit,
    backend,
    *,
    seed_transpiler: int,
    beam_seconds: float | None = None,
    line_search_seconds: float = 0.15,
    route_verifier=None,
    line_positions: tuple[int, ...] | None = None,
    return_candidate: bool = False,
) -> v1.CompileResult | tuple[v1.CompileResult, Candidate]:
    """Try complete line routes against beam search on dense and 24q+ medium QAOA."""
    shape = v1.inspect_qaoa_shape(qc)
    beam_budget = (1.0 if qc.num_qubits <= FROZEN_LINE_QUBITS else 0.0
                   ) if beam_seconds is None else float(beam_seconds)
    medium_line = (qc.num_qubits >= 24 and 0.45 <= shape.density < 0.65)
    if not (MIN_ROUTED_QUBITS <= qc.num_qubits <= MAX_ROUTED_QUBITS
            and shape.supported and (shape.density >= 0.65 or medium_line)
            and backend.num_qubits >= qc.num_qubits):
        return compile_hybrid_level3_v2(
            qc, backend, seed_transpiler=seed_transpiler,
            layouts=1, beam_seconds=beam_budget if beam_budget > 0 else 1.0,
            quality_guard=False,
        )
    if beam_budget < 0 or (qc.num_qubits <= FROZEN_LINE_QUBITS and beam_budget == 0) \
            or line_search_seconds <= 0:
        raise ValueError("Invalid beam or line search limit")

    started = perf_counter()
    cmap = backend.coupling_map
    try:
        # Keep the same 24-site prefix as the earlier baseline.
        path_length = max(FROZEN_LINE_QUBITS, qc.num_qubits)
        path = physical_line(cmap, min(path_length, cmap.size()))[:qc.num_qubits]
    except (RuntimeError, ValueError):
        path = physical_line(cmap, qc.num_qubits)
    candidates = []
    positions = (optimise_line_layout_rust(
        qc, seed=seed_transpiler, max_seconds=line_search_seconds)
        if line_positions is None else tuple(line_positions))
    if set(positions) != set(range(qc.num_qubits)):
        raise ValueError("line_positions must be a permutation of logical line sites")
    for name, layout in (
        ("line", path),
        ("line_search", tuple(path[position] for position in positions)),
    ):
        if any(layout == prior.layout for _, prior in candidates):
            continue
        line_started = perf_counter()
        candidate = Candidate(
            layout, route_line_layers_rust(qc.copy(), cmap, path, layout),
            perf_counter() - line_started,
        )
        if route_verifier is not None:
            route_verifier(name, qc, candidate.routed, candidate.layout, cmap)
        _compile_candidate(candidate, backend, seed_transpiler, shape.density)
        candidates.append((name, candidate))

    if beam_budget > 0:
        preferred = stock_initial_layout(qc, backend, seed_transpiler=seed_transpiler)
        beam_started = perf_counter()
        try:
            routed = route_beam_layers_rust(
                qc.copy(), cmap, list(preferred), max_seconds=beam_budget)
        except (v1.BeamBudgetExceeded, v1.BeamRoutingFailed):
            pass
        else:
            beam = Candidate(preferred, routed, perf_counter() - beam_started)
            if route_verifier is not None:
                route_verifier("beam", qc, beam.routed, beam.layout, cmap)
            _compile_candidate(beam, backend, seed_transpiler, shape.density)
            candidates.append(("beam", beam))

    if medium_line:
        # On medium graphs, reject a small gate saving if it adds too much depth.
        depth_limit = min(candidate.twoq_depth for _, candidate in candidates) * 1.10
        eligible = [(name, candidate) for name, candidate in candidates
                    if candidate.twoq_depth <= depth_limit]
    else:
        eligible = candidates
    name, chosen = min(eligible, key=lambda item: (
        item[1].native_2q, item[1].twoq_depth, item[1].compiled.circuit.depth()))
    counts = ", ".join(f"{label} {candidate.native_2q}/{candidate.twoq_depth}"
                       for label, candidate in candidates)
    family = f"medium{qc.num_qubits}" if medium_line else "dense"
    result = replace(
        chosen.compiled,
        route_mode=f"v2_rust_{family}_{name}",
        selector_reason=f"{family} line completion, {line_search_seconds:g}s line search, "
                        f"{beam_budget:g}s beam; candidate 2Q/depth {counts}",
        router_seconds=max(0.0, perf_counter() - started - chosen.compiled.level3_seconds),
    )
    return (result, chosen) if return_candidate else result
