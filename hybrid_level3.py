"""My QAOA-aware Level-3 variant.

The normal Qiskit Level-3 pipeline is still the fallback. The only extra idea
here is simple: detect the narrow QAOA/commuting-ZZ case I actually tested,
try the V4 beam router there, then hand the physical circuit back to Qiskit
Level 3 for the normal cleanup/translation/scheduling stages.

I intentionally do not monkey-patch Qiskit's installed package. That would make
this project harder to reproduce and easier to break. This module is the local
modified Level-3 pipeline that I benchmark against stock Qiskit Level 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from time import perf_counter
import math

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


MIN_BEAM_DENSITY = 0.45
MAX_BEAM_DENSITY = 0.80
DEFAULT_BEAM_WIDTH = 16
DEFAULT_BEAM_SECONDS = 5.0


class BeamBudgetExceeded(RuntimeError):
    """The bounded V4 search used up its time budget."""


class BeamRoutingFailed(RuntimeError):
    """The V4 search did not find a complete route inside its search space."""


@dataclass(frozen=True)
class RoutedCircuit:
    circuit: QuantumCircuit
    final_positions: list[int]
    swaps: int


@dataclass(frozen=True)
class QAOAShape:
    density: float
    supported: bool
    zz_pairs: int
    layers: int
    reason: str


@dataclass(frozen=True)
class CompileResult:
    circuit: QuantumCircuit
    final_positions: list[int]
    route_mode: str
    selector_reason: str
    density: float
    router_seconds: float
    level3_seconds: float
    swaps_before_level3: int

    @property
    def total_compile_seconds(self) -> float:
        return self.router_seconds + self.level3_seconds


@dataclass(frozen=True)
class _State:
    where: tuple[int, ...]
    pending: tuple[tuple[int, int, float], ...]
    events: tuple[tuple[str, int, int, float], ...]
    swaps: int
    steps: int


def _build_level3(target, *, initial_layout, seed_transpiler: int):
    return generate_preset_pass_manager(
        optimization_level=3,
        target=target,
        initial_layout=initial_layout,
        routing_method="sabre",
        seed_transpiler=int(seed_transpiler),
        approximation_degree=1.0,
        qubits_initially_zero=False,
        scheduling_method="alap",
    )


def _final_positions(compiled: QuantumCircuit, logical_qubits: int) -> list[int]:
    positions = list(compiled.layout.final_index_layout())
    if len(positions) < logical_qubits:
        raise ValueError("Qiskit returned fewer final layout entries than logical qubits.")
    positions = [int(q) for q in positions[:logical_qubits]]
    if len(set(positions)) != logical_qubits:
        raise ValueError("Qiskit returned duplicate physical output positions.")
    return positions


def inspect_qaoa_shape(qc: QuantumCircuit) -> QAOAShape:
    """Recognize the exact QAOA shape V4 was designed and tested for."""
    # I am deliberately strict here. If the circuit is not the kind I actually tested,
    # the hybrid should be boring and let normal Qiskit handle it.
    pairs = set()
    first_layer_pairs = None
    layers = 0
    supported = True
    reason = "supported commuting-ZZ QAOA"

    for instruction in qc.data:
        gate = instruction.operation
        if isinstance(gate, PauliEvolutionGate):
            layers += 1
            if gate.num_qubits != qc.num_qubits or not isinstance(gate.operator, SparsePauliOp):
                supported = False
                reason = "evolution layer does not cover the full logical circuit"
                continue

            layer = {}
            for label, qubits, coefficient in gate.operator.to_sparse_list():
                try:
                    value = complex(coefficient)
                except (TypeError, ValueError):
                    supported = False
                    reason = "non-numeric ZZ coefficient"
                    break
                if (
                    label != "ZZ"
                    or len(qubits) != 2
                    or not math.isfinite(value.real)
                    or not math.isfinite(value.imag)
                    or abs(value.imag) > 1e-10
                ):
                    supported = False
                    reason = "evolution layer contains something other than real two-qubit ZZ terms"
                    break
                a, b = sorted(qc.find_bit(instruction.qubits[i]).index for i in qubits)
                layer[a, b] = layer.get((a, b), 0.0) + value.real

            layer_pairs = {pair for pair, weight in layer.items() if abs(weight) > 1e-12}
            if len(layer_pairs) != len(layer):
                supported = False
                reason = "a ZZ pair cancels inside one layer"
            if first_layer_pairs is None:
                first_layer_pairs = layer_pairs
            elif layer_pairs != first_layer_pairs:
                supported = False
                reason = "different ZZ interaction graphs across QAOA layers"
            pairs.update(layer_pairs)
            continue

        if not isinstance(gate, Gate) or gate.num_qubits != 1 or instruction.clbits:
            supported = False
            reason = "unsupported gate outside the ZZ evolution layers"

    if layers == 0:
        supported = False
        reason = "no PauliEvolutionGate ZZ layer was found"
    if qc.num_clbits:
        supported = False
        reason = "classical bits are present before routing"

    possible = qc.num_qubits * (qc.num_qubits - 1) // 2
    density = len(pairs) / possible if possible else 0.0
    return QAOAShape(density, supported, len(pairs), layers, reason)


def _chip_data(cmap: CouplingMap):
    physical = tuple(sorted(cmap.physical_qubits))
    neighbors = {q: set() for q in physical}
    for a, b in cmap.get_edges():
        neighbors[a].add(b)
        neighbors[b].add(a)

    distances = {}
    paths = {}
    for start in physical:
        queue = [start]
        previous = {start: None}
        for current in queue:
            for nxt in sorted(neighbors[current]):
                if nxt not in previous:
                    previous[nxt] = current
                    queue.append(nxt)
        for finish in physical:
            if finish not in previous:
                raise ValueError("The hardware coupling map must be connected.")
            path = []
            current = finish
            while current != start:
                parent = previous[current]
                path.append(tuple(sorted((parent, current))))
                current = parent
            path.reverse()
            distances[start, finish] = len(path)
            paths[start, finish] = tuple(path)
    return physical, neighbors, distances, paths


def _pending_dict(values):
    return {(u, v): angle for u, v, angle in values}


def _normalise(state: _State, distances):
    pending = _pending_dict(state.pending)
    events = list(state.events)
    for pair in sorted(tuple(pending)):
        u, v = pair
        if distances[state.where[u], state.where[v]] == 1:
            events.append(("rzz", state.where[u], state.where[v], pending.pop(pair)))
    return _State(
        where=state.where,
        pending=tuple((u, v, pending[u, v]) for u, v in sorted(pending)),
        events=tuple(events),
        swaps=state.swaps,
        steps=state.steps,
    )


def _apply_batch(state: _State, batch, at):
    where = list(state.where)
    occupied = list(at)
    for x, y in batch:
        left, right = occupied[x], occupied[y]
        occupied[x], occupied[y] = right, left
        if left is not None:
            where[left] = y
        if right is not None:
            where[right] = x
    events = state.events + tuple(("swap", x, y, 0.0) for x, y in batch)
    return (
        _State(
            tuple(where),
            state.pending,
            events,
            state.swaps + len(batch),
            state.steps + 1,
        ),
        occupied,
    )


def _focus_edges(state: _State, at, edges, distances):
    pending = list(state.pending)
    urgency = sorted(
        pending,
        key=lambda item: (
            distances[state.where[item[0]], state.where[item[1]]],
            sum(1 for a, b, _ in pending if item[0] in (a, b) or item[1] in (a, b)),
        ),
        reverse=True,
    )
    focus = urgency[: min(6, len(urgency))]
    candidates = set()
    for u, v, _ in focus:
        a, b = state.where[u], state.where[v]
        old = distances[a, b]
        for x, y in edges:
            if x not in (a, b) and y not in (a, b):
                continue
            moved = y if x == a else x if y == a else a
            other = b
            if x == b:
                moved, other = y, a
            elif y == b:
                moved, other = x, a
            if distances[moved, other] < old:
                candidates.add(tuple(sorted((x, y))))
    if not candidates:
        candidates = set(edges)

    def edge_gain(edge):
        x, y = edge
        gain = 0
        for u, v, _ in pending:
            before = distances[state.where[u], state.where[v]]
            after_u = y if at[x] == u else x if at[y] == u else state.where[u]
            after_v = y if at[x] == v else x if at[y] == v else state.where[v]
            gain += before - distances[after_u, after_v]
        return gain

    return tuple(sorted(candidates, key=lambda edge: (edge_gain(edge), edge), reverse=True))


def _batches(state: _State, at, edges, distances, max_candidates=24):
    # Trying every possible SWAP batch explodes way too fast. I keep a small shortlist
    # of useful-looking single swaps, then build a few non-overlapping pairs/triples.
    singles = list(_focus_edges(state, at, edges, distances))[:10]
    batches = [(edge,) for edge in singles]
    for size in (2, 3):
        for batch in combinations(singles[:8], size):
            sites = [site for edge in batch for site in edge]
            if len(set(sites)) == len(sites):
                batches.append(tuple(sorted(batch)))
    return tuple(batches[:max_candidates])


def _score(state: _State, distances, paths, degree):
    pending = list(state.pending)
    if not pending:
        return (state.swaps, state.steps, 0.0)

    # Count all remaining work first. Doing this inside the scoring loop used to make
    # the score depend on the order of ZZ pairs, which was a pretty dumb hidden bug.
    remaining = [0] * len(state.where)
    for u, v, _ in pending:
        remaining[u] += 1
        remaining[v] += 1

    edge_load = {}
    distance_cost = 0.0
    max_distance = 0
    for u, v, _ in pending:
        distance = distances[state.where[u], state.where[v]]
        distance_cost += max(0, distance - 1) * (1.0 + 0.03 * (remaining[u] + remaining[v]))
        max_distance = max(max_distance, distance)
        for edge in paths[state.where[u], state.where[v]]:
            edge_load[edge] = edge_load.get(edge, 0) + 1

    congestion = max(edge_load.values(), default=0) + 0.15 * sum(
        value * value for value in edge_load.values()
    )
    trap_penalty = sum(
        count / max(1, degree[state.where[logical]])
        for logical, count in enumerate(remaining)
    )
    return (
        4.0 * distance_cost
        + 1.5 * max_distance
        + 0.8 * congestion
        + 0.4 * trap_penalty
        + 0.25 * state.swaps
        + 0.15 * state.steps,
        state.swaps,
        state.steps,
    )


def _route_layer(pending, where, cmap, *, beam_width, deadline):
    physical, neighbors, distances, paths = _chip_data(cmap)
    edges = tuple(sorted(tuple(sorted(edge)) for edge in cmap.get_edges()))
    degree = {site: len(neighbors[site]) for site in physical}
    initial = _normalise(
        _State(
            tuple(where),
            tuple((u, v, angle) for (u, v), angle in sorted(pending.items())),
            tuple(),
            0,
            0,
        ),
        distances,
    )
    beam = [initial]
    max_steps = max(8, len(pending) * (len(physical) + 1))

    for _ in range(max_steps):
        if deadline is not None and perf_counter() >= deadline:
            raise BeamBudgetExceeded
        next_states = {}
        finished = []

        for state in beam:
            if deadline is not None and perf_counter() >= deadline:
                raise BeamBudgetExceeded
            if not state.pending:
                finished.append(state)
                continue

            current_at = [None] * len(physical)
            for logical, site in enumerate(state.where):
                current_at[site] = logical

            for batch in _batches(state, current_at, edges, distances):
                candidate, _ = _apply_batch(state, batch, current_at)
                candidate = _normalise(candidate, distances)
                signature = (
                    candidate.where,
                    tuple((u, v) for u, v, _ in candidate.pending),
                )
                old = next_states.get(signature)
                if old is None or _score(candidate, distances, paths, degree) < _score(
                    old, distances, paths, degree
                ):
                    next_states[signature] = candidate

        if finished:
            best = min(finished, key=lambda state: (state.swaps, state.steps))
            return list(best.events), list(best.where), best.swaps
        if not next_states:
            break
        beam = sorted(
            next_states.values(),
            key=lambda state: _score(state, distances, paths, degree),
        )[:beam_width]

    raise BeamRoutingFailed("V4 could not route the whole ZZ layer.")


def _layer_terms(qc, instruction):
    gate = instruction.operation
    if gate.num_qubits != qc.num_qubits or not isinstance(gate.operator, SparsePauliOp):
        raise ValueError("V4 needs a full-width SparsePauliOp ZZ layer.")
    terms = {}
    for label, qubits, coefficient in gate.operator.to_sparse_list():
        value = complex(coefficient)
        if label != "ZZ" or len(qubits) != 2 or abs(value.imag) > 1e-10:
            raise ValueError("V4 only supports real two-qubit ZZ terms.")
        u, v = sorted(qc.find_bit(instruction.qubits[i]).index for i in qubits)
        terms[u, v] = terms.get((u, v), 0.0) + 2 * value.real * gate.time
    return terms


def route_beam_layers(
    qc: QuantumCircuit,
    cmap: CouplingMap,
    initial_layout: list[int],
    *,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    max_seconds: float | None = DEFAULT_BEAM_SECONDS,
) -> RoutedCircuit:
    """Route a supported commuting-ZZ QAOA circuit with V4 beam search."""
    n = qc.num_qubits
    chip_size = cmap.size()
    if len(initial_layout) != n or len(set(initial_layout)) != n:
        raise ValueError("Initial layout needs one different physical qubit per logical qubit.")
    if not set(initial_layout).issubset(cmap.physical_qubits):
        raise ValueError("Initial layout contains a qubit that is not on this chip.")

    physical = QuantumCircuit(chip_size, global_phase=qc.global_phase)
    where = list(initial_layout)
    at = [None] * chip_size
    for logical, site in enumerate(where):
        at[site] = logical

    total_swaps = 0
    deadline = perf_counter() + max_seconds if max_seconds is not None else None
    for instruction in qc.data:
        gate = instruction.operation
        if isinstance(gate, PauliEvolutionGate):
            events, where, swaps = _route_layer(
                _layer_terms(qc, instruction),
                where,
                cmap,
                beam_width=beam_width,
                deadline=deadline,
            )
            for kind, a, b, angle in events:
                if kind == "rzz":
                    physical.rzz(angle, a, b)
                else:
                    physical.swap(a, b)
            at = [None] * chip_size
            for logical, site in enumerate(where):
                at[site] = logical
            total_swaps += swaps
            continue

        if gate.num_qubits != 1 or instruction.clbits:
            raise ValueError(f"Unsupported gate outside a ZZ layer: {gate.name}")
        logical = qc.find_bit(instruction.qubits[0]).index
        physical.append(gate, [where[logical]])

    return RoutedCircuit(physical, where, total_swaps)


def compile_stock_level3(qc: QuantumCircuit, backend, *, seed_transpiler: int) -> CompileResult:
    """Compile the exact input with normal Qiskit optimization_level=3."""
    if isinstance(seed_transpiler, bool) or not isinstance(seed_transpiler, Integral):
        raise ValueError("seed_transpiler must be an integer.")
    start = perf_counter()
    manager = _build_level3(backend.target, initial_layout=None, seed_transpiler=int(seed_transpiler))
    compiled = manager.run(qc)
    elapsed = perf_counter() - start
    return CompileResult(
        circuit=compiled,
        final_positions=_final_positions(compiled, qc.num_qubits),
        route_mode="stock_qiskit",
        selector_reason="stock Qiskit Level 3 baseline",
        density=inspect_qaoa_shape(qc).density,
        router_seconds=0.0,
        level3_seconds=elapsed,
        swaps_before_level3=0,
    )


def compile_hybrid_level3(
    qc: QuantumCircuit,
    backend,
    *,
    seed_transpiler: int,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    beam_seconds: float | None = DEFAULT_BEAM_SECONDS,
) -> CompileResult:
    """Run my local modified Level-3 pipeline with an automatic safe fallback."""
    if isinstance(seed_transpiler, bool) or not isinstance(seed_transpiler, Integral):
        raise ValueError("seed_transpiler must be an integer.")

    shape = inspect_qaoa_shape(qc)
    cmap = backend.coupling_map
    tested_shape = qc.num_qubits == 16 and backend.num_qubits == 16
    # This selector is intentionally simple for V1. Medium/dense ZZ graphs are where
    # V4 has evidence behind it; everything else stays on stock Qiskit instead of gambling.
    use_beam = (
        tested_shape
        and shape.supported
        and MIN_BEAM_DENSITY <= shape.density <= MAX_BEAM_DENSITY
    )

    if not use_beam:
        if not tested_shape:
            reason = "outside the tested 16-logical/16-physical-qubit setup"
        elif not shape.supported:
            reason = shape.reason
        elif shape.density < MIN_BEAM_DENSITY:
            reason = f"density {shape.density:.3f} is below the V4 window"
        else:
            reason = f"density {shape.density:.3f} is above the V4 window"
        stock = compile_stock_level3(qc, backend, seed_transpiler=int(seed_transpiler))
        return CompileResult(
            circuit=stock.circuit,
            final_positions=stock.final_positions,
            route_mode="qiskit_fallback",
            selector_reason=reason,
            density=shape.density,
            router_seconds=0.0,
            level3_seconds=stock.level3_seconds,
            swaps_before_level3=0,
        )

    initial_layout = list(range(qc.num_qubits))
    route_start = perf_counter()
    try:
        routed = route_beam_layers(
            qc.copy(),
            cmap,
            initial_layout,
            beam_width=beam_width,
            max_seconds=beam_seconds,
        )
    except (BeamBudgetExceeded, BeamRoutingFailed, ValueError) as exc:
        # Failing safely matters more than forcing V4 to finish. A bad search just hands
        # the original circuit back to normal Level 3 and records why it happened.
        route_seconds = perf_counter() - route_start
        stock = compile_stock_level3(qc, backend, seed_transpiler=int(seed_transpiler))
        return CompileResult(
            circuit=stock.circuit,
            final_positions=stock.final_positions,
            route_mode="qiskit_fallback",
            selector_reason=f"V4 fallback: {type(exc).__name__}",
            density=shape.density,
            router_seconds=route_seconds,
            level3_seconds=stock.level3_seconds,
            swaps_before_level3=0,
        )
    route_seconds = perf_counter() - route_start

    level3_start = perf_counter()
    manager = _build_level3(
        backend.target,
        initial_layout=list(range(backend.num_qubits)),
        seed_transpiler=int(seed_transpiler),
    )
    compiled = manager.run(routed.circuit)
    level3_seconds = perf_counter() - level3_start

    physical_output = _final_positions(compiled, backend.num_qubits)
    final_positions = [physical_output[q] for q in routed.final_positions]
    if len(set(final_positions)) != qc.num_qubits:
        raise ValueError("Hybrid final-position composition produced duplicate physical qubits.")

    return CompileResult(
        circuit=compiled,
        final_positions=final_positions,
        route_mode="v4_beam",
        selector_reason=(
            f"supported QAOA with density {shape.density:.3f} inside "
            f"[{MIN_BEAM_DENSITY:.2f}, {MAX_BEAM_DENSITY:.2f}]"
        ),
        density=shape.density,
        router_seconds=route_seconds,
        level3_seconds=level3_seconds,
        swaps_before_level3=routed.swaps,
    )
