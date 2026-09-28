"""Rust beam-search adapter and 16-qubit fallback used by Hybrid v2.0.0."""
from __future__ import annotations

from numbers import Integral
from time import perf_counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate

import hybrid_level3 as v1

try:
    from qaoa_hybrid import qaoa_v2_rust
except ImportError as exc:  # Give a useful error instead of silently falling back to Python.
    try:
        import qaoa_v2_rust
    except ImportError:
        raise ImportError(
            "The Rust router is missing. Install the project with pip, "
            "or build the research core with build_rust.ps1."
        ) from exc


RUST_CORE_VERSION = getattr(qaoa_v2_rust, "__version__", "unknown")


def _rust_route_layer(
    pending: dict[tuple[int, int], float],
    where: list[int],
    cmap,
    *,
    beam_width: int,
    max_seconds: float | None,
):
    """Call the Rust port of V1 _route_layer with the same logical inputs."""
    # V1 starts each layer from sorted(pending.items()). Make that ordering explicit at
    # the language boundary so the Rust port never depends on Python dict insertion order.
    pending_items = [
        (int(u), int(v), float(angle))
        for (u, v), angle in sorted(pending.items())
    ]
    edges = [(int(a), int(b)) for a, b in cmap.get_edges()]

    try:
        events, final_where, swaps = qaoa_v2_rust.route_layer(
            pending_items,
            [int(site) for site in where],
            int(cmap.size()),
            edges,
            int(beam_width),
            max_seconds,
        )
    except TimeoutError as exc:
        raise v1.BeamBudgetExceeded from exc
    except RuntimeError as exc:
        raise v1.BeamRoutingFailed(str(exc)) from exc

    return (
        [(str(kind), int(a), int(b), float(angle)) for kind, a, b, angle in events],
        [int(site) for site in final_where],
        int(swaps),
    )


def route_beam_layers_rust(
    qc: QuantumCircuit,
    cmap,
    initial_layout: list[int],
    *,
    beam_width: int = v1.DEFAULT_BEAM_WIDTH,
    max_seconds: float | None = v1.DEFAULT_BEAM_SECONDS,
) -> v1.RoutedCircuit:
    """Route the V1-supported QAOA shape using the 1:1 Rust Hybrid search core."""
    n = qc.num_qubits
    chip_size = cmap.size()
    if len(initial_layout) != n or len(set(initial_layout)) != n:
        raise ValueError("Initial layout needs one different physical qubit per logical qubit.")
    if not set(initial_layout).issubset(cmap.physical_qubits):
        raise ValueError("Initial layout contains a qubit that is not on this chip.")

    physical = QuantumCircuit(chip_size, global_phase=qc.global_phase)
    where = list(initial_layout)
    total_swaps = 0

    # V1 has one deadline for the whole circuit, not a new 5 s budget per QAOA layer.
    # Rust receives the remaining time before each layer so p=2 keeps that same policy.
    deadline = perf_counter() + max_seconds if max_seconds is not None else None

    for instruction in qc.data:
        gate = instruction.operation
        if isinstance(gate, PauliEvolutionGate):
            remaining_seconds = None if deadline is None else deadline - perf_counter()
            events, where, swaps = _rust_route_layer(
                v1._layer_terms(qc, instruction),
                where,
                cmap,
                beam_width=beam_width,
                max_seconds=remaining_seconds,
            )
            for kind, a, b, angle in events:
                if kind == "rzz":
                    physical.rzz(angle, a, b)
                elif kind == "swap":
                    physical.swap(a, b)
                else:
                    raise ValueError(f"Rust Hybrid returned an unknown event kind: {kind}")
            total_swaps += swaps
            continue

        if gate.num_qubits != 1 or instruction.clbits:
            raise ValueError(f"Unsupported gate outside a ZZ layer: {gate.name}")
        logical = qc.find_bit(instruction.qubits[0]).index
        physical.append(gate, [where[logical]])

    return v1.RoutedCircuit(physical, where, total_swaps)


def route_line_layers_rust(
    qc: QuantumCircuit,
    cmap,
    path: tuple[int, ...],
    initial_layout: tuple[int, ...] | None = None,
) -> v1.RoutedCircuit:
    """Complete each commuting ZZ layer with Rust's line SWAP schedule."""
    n = qc.num_qubits
    if len(path) != n or len(set(path)) != n:
        raise ValueError("The physical line must have one site per logical qubit")
    where = list(initial_layout if initial_layout is not None else path)
    if len(where) != n or set(where) != set(path):
        raise ValueError("The starting layout must occupy the physical line")

    physical = QuantumCircuit(cmap.size(), global_phase=qc.global_phase)
    edges = [(int(a), int(b)) for a, b in cmap.get_edges()]
    total_swaps = 0
    for instruction in qc.data:
        gate = instruction.operation
        if isinstance(gate, PauliEvolutionGate):
            pending = [(int(u), int(v), float(angle))
                       for (u, v), angle in sorted(v1._layer_terms(qc, instruction).items())]
            events, where, swaps = qaoa_v2_rust.route_line_layer(
                pending, where, int(cmap.size()), edges, list(path))
            for kind, a, b, angle in events:
                if kind == "rzz":
                    physical.rzz(float(angle), int(a), int(b))
                elif kind == "swap":
                    physical.swap(int(a), int(b))
                else:
                    raise ValueError(f"Rust line route returned an unknown event: {kind}")
            total_swaps += int(swaps)
        else:
            if gate.num_qubits != 1 or instruction.clbits:
                raise ValueError(f"Unsupported gate outside a ZZ layer: {gate.name}")
            logical = qc.find_bit(instruction.qubits[0]).index
            physical.append(gate, [where[logical]])
    return v1.RoutedCircuit(physical, where, total_swaps)


def optimise_line_layout_rust(
    qc: QuantumCircuit, *, seed: int = 11, max_seconds: float = 0.15,
) -> tuple[int, ...]:
    """Search for a logical ordering that finishes the line schedule earlier."""
    layer = next((inst for inst in qc.data if isinstance(inst.operation, PauliEvolutionGate)), None)
    if layer is None:
        raise ValueError("No ZZ layer to place on the line")
    edges = sorted(v1._layer_terms(qc, layer))
    positions = tuple(qaoa_v2_rust.optimise_line_layout(
        qc.num_qubits, edges, int(seed), float(max_seconds)))
    if set(positions) != set(range(qc.num_qubits)):
        raise AssertionError("Rust returned an invalid line placement")
    return positions


def compile_rust_core_level3(
    qc: QuantumCircuit,
    backend,
    *,
    seed_transpiler: int,
    beam_width: int = v1.DEFAULT_BEAM_WIDTH,
    beam_seconds: float | None = v1.DEFAULT_BEAM_SECONDS,
) -> v1.CompileResult:
    """Run the V1 Hybrid policy with only the Hybrid search core replaced by Rust."""
    if isinstance(seed_transpiler, bool) or not isinstance(seed_transpiler, Integral):
        raise ValueError("seed_transpiler must be an integer.")

    shape = v1.inspect_qaoa_shape(qc)
    cmap = backend.coupling_map
    tested_shape = qc.num_qubits == 16 and backend.num_qubits == 16
    use_beam = (
        tested_shape
        and shape.supported
        and v1.MIN_BEAM_DENSITY <= shape.density <= v1.MAX_BEAM_DENSITY
    )

    if not use_beam:
        if not tested_shape:
            reason = "outside the tested 16-logical/16-physical-qubit setup"
        elif not shape.supported:
            reason = shape.reason
        elif shape.density < v1.MIN_BEAM_DENSITY:
            reason = f"density {shape.density:.3f} is below the Hybrid window"
        else:
            reason = f"density {shape.density:.3f} is above the Hybrid window"

        stock = v1.compile_stock_level3(qc, backend, seed_transpiler=int(seed_transpiler))
        return v1.CompileResult(
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
        routed = route_beam_layers_rust(
            qc.copy(),
            cmap,
            initial_layout,
            beam_width=beam_width,
            max_seconds=beam_seconds,
        )
    except (v1.BeamBudgetExceeded, v1.BeamRoutingFailed, ValueError) as exc:
        route_seconds = perf_counter() - route_start
        stock = v1.compile_stock_level3(qc, backend, seed_transpiler=int(seed_transpiler))
        return v1.CompileResult(
            circuit=stock.circuit,
            final_positions=stock.final_positions,
            route_mode="qiskit_fallback",
            selector_reason=f"Hybrid Rust fallback: {type(exc).__name__}",
            density=shape.density,
            router_seconds=route_seconds,
            level3_seconds=stock.level3_seconds,
            swaps_before_level3=0,
        )
    route_seconds = perf_counter() - route_start

    level3_start = perf_counter()
    manager = v1._build_level3(
        backend.target,
        initial_layout=list(range(backend.num_qubits)),
        seed_transpiler=int(seed_transpiler),
    )
    compiled = manager.run(routed.circuit)
    level3_seconds = perf_counter() - level3_start

    physical_output = v1._final_positions(compiled, backend.num_qubits)
    final_positions = [physical_output[q] for q in routed.final_positions]
    if len(set(final_positions)) != qc.num_qubits:
        raise ValueError("Hybrid final-position composition produced duplicate physical qubits.")

    return v1.CompileResult(
        circuit=compiled,
        final_positions=final_positions,
        route_mode="v2_beam_rust",
        selector_reason=(
            f"supported QAOA with density {shape.density:.3f} inside "
            f"[{v1.MIN_BEAM_DENSITY:.2f}, {v1.MAX_BEAM_DENSITY:.2f}]"
        ),
        density=shape.density,
        router_seconds=route_seconds,
        level3_seconds=level3_seconds,
        swaps_before_level3=routed.swaps,
    )
