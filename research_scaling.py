"""Memory-aware scaling study for Stock Qiskit Level 3 and Hybrid v2.0.0.

The full 127-site fake backend is used at every logical size. A route is checked
symbolically at every size by replaying its SWAPs and comparing the resulting
logical gate stream (commuting ZZ terms within each QAOA layer may be reordered).
Optional exact state checks stop at 20 logical qubits; a 2**24 state is never built.
These are local compiler results, not QPU measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKyiv, FakeSherbrooke

import benchmark as bench
import hybrid_level3 as v1
from hybrid_level3_v2 import LayoutSearchFailed, compare_variants


ROOT = Path(__file__).resolve().parent
BACKENDS = {cls.__name__: cls for cls in (FakeBrisbane, FakeKyiv, FakeSherbrooke)}
FAMILIES = {"medium50": 0.50, "dense75": 0.75}
FIELDS = (
    "snapshot", "logical_qubits", "family", "graph_id", "layers", "edges",
    "stock_2q", "raw_2q", "guarded_2q", "stock_2q_depth", "raw_2q_depth",
    "guarded_2q_depth", "stock_seconds", "raw_total_seconds",
    "guarded_total_seconds", "routing_status", "routes_verified",
    "max_route_state_error", "mps_stock_probability_error",
    "mps_raw_probability_error", "peak_process_rss_mib",
)


def process_memory_mib() -> tuple[float | None, float | None]:
    """Return current and peak process RSS; do not require an extra package."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        size_t = ctypes.c_size_t

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", size_t), ("WorkingSetSize", size_t),
                        ("QuotaPeakPagedPoolUsage", size_t), ("QuotaPagedPoolUsage", size_t),
                        ("QuotaPeakNonPagedPoolUsage", size_t), ("QuotaNonPagedPoolUsage", size_t),
                        ("PagefileUsage", size_t), ("PeakPagefileUsage", size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        scale = 1024 * 1024
        return counters.WorkingSetSize / scale, counters.PeakWorkingSetSize / scale
    try:
        import resource
        peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return None, peak_kib / 1024 if sys.platform != "darwin" else peak_kib / (1024 * 1024)
    except ImportError:
        return None, None


def graph_edges(n: int, density: float, seed: int) -> tuple[tuple[int, int], ...]:
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    count = round(len(possible) * density)
    picked = np.random.default_rng(seed).choice(len(possible), count, replace=False)
    return tuple(sorted(possible[int(index)] for index in picked))


def qaoa_circuit(n: int, edges: tuple[tuple[int, int], ...], layers: int,
                 seed: int) -> QuantumCircuit:
    angles = np.random.default_rng(seed).uniform(0.15, 0.95, size=(layers, 2))
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    operator = SparsePauliOp.from_sparse_list(
        [("ZZ", [a, b], 1.0) for a, b in edges], num_qubits=n)
    for gamma, beta in angles:
        circuit.append(PauliEvolutionGate(operator, time=float(gamma)), range(n))
        circuit.rx(2 * float(beta), range(n))
    return circuit


def gate_token(gate, logical: int) -> tuple:
    return ("single", logical, gate.name, tuple(round(float(value), 12) for value in gate.params))


def route_replay(reference: QuantumCircuit, routed, layout: tuple[int, ...],
                 cmap) -> QuantumCircuit:
    """Check the complete logical operation stream after physical SWAP replay."""
    n = reference.num_qubits
    physical_count = routed.circuit.num_qubits
    if len(layout) != n or len(set(layout)) != n:
        raise AssertionError("Invalid starting layout")
    edges = {tuple(sorted((a, b))) for a, b in cmap.get_edges()}
    expected = []
    logical_reference = QuantumCircuit(n, global_phase=reference.global_phase)
    for instruction in reference.data:
        gate = instruction.operation
        if isinstance(gate, PauliEvolutionGate):
            terms = v1._layer_terms(reference, instruction)
            expected.append(("zz", Counter(
                (a, b, round(float(angle), 12)) for (a, b), angle in terms.items())))
            for (a, b), angle in sorted(terms.items()):
                logical_reference.rzz(float(angle), a, b)
        else:
            logical = reference.find_bit(instruction.qubits[0]).index
            expected.append(gate_token(gate, logical))
            logical_reference.append(gate, [logical])

    at = [None] * physical_count
    for logical, physical in enumerate(layout):
        at[physical] = logical
    actual = []
    pending = Counter()
    replay = QuantumCircuit(n, global_phase=routed.circuit.global_phase)
    swap_count = 0
    for instruction in routed.circuit.data:
        gate = instruction.operation
        sites = tuple(routed.circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if gate.name == "swap":
            if tuple(sorted(sites)) not in edges:
                raise AssertionError(f"SWAP on a disconnected physical pair: {sites}")
            a, b = sites
            at[a], at[b] = at[b], at[a]
            swap_count += 1
        elif gate.name == "rzz":
            if tuple(sorted(sites)) not in edges:
                raise AssertionError(f"RZZ on a disconnected physical pair: {sites}")
            a, b = (at[site] for site in sites)
            if a is None or b is None:
                raise AssertionError("RZZ acted on an unoccupied site")
            pending[min(a, b), max(a, b), round(float(gate.params[0]), 12)] += 1
            replay.rzz(float(gate.params[0]), a, b)
        elif gate.num_qubits == 1 and not instruction.clbits:
            if pending:
                actual.append(("zz", pending))
                pending = Counter()
            logical = at[sites[0]]
            if logical is None:
                raise AssertionError("One-qubit gate acted on an unoccupied site")
            actual.append(gate_token(gate, logical))
            replay.append(gate, [logical])
        else:
            raise AssertionError(f"Unexpected routed operation: {gate.name}")
    if pending:
        actual.append(("zz", pending))
    if actual != expected:
        raise AssertionError("Routed logical gate stream differs from QAOA input")
    if tuple(at.index(logical) for logical in range(n)) != tuple(routed.final_positions):
        raise AssertionError("Router final positions disagree with SWAP replay")
    if swap_count != routed.swaps:
        raise AssertionError("Router SWAP count disagrees with routed circuit")
    if abs(float(replay.global_phase - logical_reference.global_phase)) > 1e-12:
        raise AssertionError("Router changed the global phase")
    return replay


def state_error(reference: QuantumCircuit, replay: QuantumCircuit) -> float:
    """Small-n numerical check; both inputs contain only 1Q gates and RZZ."""
    ideal = Statevector.from_instruction(reference).data
    actual = Statevector.from_instruction(replay).data
    return bench.amplitude_error(ideal, actual)


def native_metrics(compiled, backend) -> tuple[int, int]:
    circuit = compiled.circuit
    if len(compiled.final_positions) != len(set(compiled.final_positions)):
        raise AssertionError("Compiled final positions are not unique")
    count = 0
    for instruction in circuit.data:
        if instruction.operation.num_qubits != 2:
            continue
        sites = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if not backend.target.instruction_supported(
                operation_name=instruction.operation.name, qargs=sites):
            raise AssertionError(f"Compiled gate {instruction.operation.name} is not native at {sites}")
        count += 1
    return count, bench.twoq_depth(circuit)


def mps_probability_error(compiled, ideal: np.ndarray, simulator) -> float:
    circuit = compiled.circuit.copy()
    circuit.save_probabilities(qubits=compiled.final_positions, label="logical_probabilities")
    result = simulator.run(circuit, shots=1).result()
    if not result.success:
        raise RuntimeError(f"MPS simulation failed: {result.status}")
    error = float(np.max(np.abs(result.data()["logical_probabilities"] - ideal)))
    if error > 1e-8:
        raise AssertionError(f"Compiled probability mismatch: {error}")
    return error


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def paired_line(label: str, group: list[dict]) -> str:
    base = sum(int(row["stock_2q"]) for row in group)
    trial = sum(int(row["raw_2q"]) for row in group)
    wins = sum(int(row["raw_2q"]) < int(row["stock_2q"]) for row in group)
    ties = sum(int(row["raw_2q"]) == int(row["stock_2q"]) for row in group)
    losses = len(group) - wins - ties
    routed = sum(row["routing_status"] == "routed" for row in group)
    change = f"{(trial / base - 1) * 100:+.1f}%" if base else "n/a"
    return (f"| {label} | {len(group)} | {routed} | {base} | {trial} | "
            f"{change} | {wins}/{ties}/{losses} | "
            f"{median(float(row['stock_seconds']) for row in group):.3f} | "
            f"{median(float(row['raw_total_seconds']) for row in group):.3f} |")


def write_report(path: Path, rows: list[dict], args) -> None:
    lines = ["# Logical-qubit scaling on full IBM fake targets", "",
             "Local compiler benchmark only. Each case uses the complete 127-physical-qubit "
             "fake target; no IBM QPU was run. Hybrid starts from Stock Qiskit's initial "
             "layout, so Hybrid time includes that Stock compilation. Raw Hybrid uses a "
             "Stock fallback when Rust cannot complete routing; guarded additionally picks "
             "the lower-native-2Q result between Stock and raw Hybrid.", "",
             f"Graph seed base: {args.graph_start}; graph instances per family: {args.graphs}; "
             f"search limit per layout: {args.beam_seconds:g} s; layout starts: {args.layouts}; "
             f"beam width: {args.beam_width}; "
             f"compiler seed: {args.seed}.",
             f"Snapshots: {', '.join(args.backends)}. Graph families: "
             f"{', '.join(args.families)}; QAOA layers: {', '.join(map(str, args.layers))}.",
             "At every size, route correctness is checked by replaying SWAPs and comparing "
             "the logical gate stream, allowing only ZZ reordering within a commuting layer. "
             f"Numerical exact-state checks run through {args.statevector_up_to} logical qubits. "
             "No 2^24 statevector is constructed. Optional compiled-circuit MPS probability "
             "checks cover selected 16-qubit cases only; probabilities do not prove full phase equivalence.", "",
             "| Logical size | Cases | Rust routes | Stock 2Q | Raw Hybrid 2Q | Change | Raw wins/ties/losses | Median Stock s | Median Hybrid s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in sorted({int(row["logical_qubits"]) for row in rows}):
        group = [row for row in rows if int(row["logical_qubits"]) == n]
        lines.append(paired_line(str(n), group))
    lines.extend(["", "## Guarded portfolio output", "",
                  "Guarded compares the raw Hybrid circuit with the Stock circuit already "
                  "compiled for its initial layout. These gains include that Stock choice.", "",
                  "| Logical size | Cases | Stock 2Q | Guarded 2Q | Change | Guarded wins/ties/losses |",
                  "|---|---:|---:|---:|---:|---:|"])
    for n in sorted({int(row["logical_qubits"]) for row in rows}):
        group = [row for row in rows if int(row["logical_qubits"]) == n]
        base = sum(int(row["stock_2q"]) for row in group)
        guard = sum(int(row["guarded_2q"]) for row in group)
        wins = sum(int(row["guarded_2q"]) < int(row["stock_2q"]) for row in group)
        ties = sum(int(row["guarded_2q"]) == int(row["stock_2q"]) for row in group)
        lines.append(f"| {n} | {len(group)} | {base} | {guard} | "
                     f"{(guard / base - 1) * 100:+.1f}% | {wins}/{ties}/{len(group) - wins - ties} |")
    lines.extend(["", "## By graph family and layer count", "",
                  "The same graph/layer workload appears on each selected snapshot.", "",
                  "| Logical size | Family | Layers | Paired cases | Rust routes | Raw 2Q change | Guarded 2Q change |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for n in sorted({int(row["logical_qubits"]) for row in rows}):
        for family in args.families:
            for depth in args.layers:
                group = [row for row in rows if int(row["logical_qubits"]) == n
                         and row["family"] == family and int(row["layers"]) == depth]
                if not group:
                    continue
                base = sum(int(row["stock_2q"]) for row in group)
                raw = sum(int(row["raw_2q"]) for row in group)
                guard = sum(int(row["guarded_2q"]) for row in group)
                routed = sum(row["routing_status"] == "routed" for row in group)
                lines.append(f"| {n} | {family} | {depth} | {len(group)} | {routed} | "
                             f"{(raw / base - 1) * 100:+.1f}% | "
                             f"{(guard / base - 1) * 100:+.1f}% |")
    lines.extend(["", "## Successful Rust routes only", "",
                  "The next rows exclude Stock fallback cases on both sides.", "",
                  "| Logical size | Paired routes | Stock 2Q | Raw Hybrid 2Q | Change | Wins/ties/losses |",
                  "|---|---:|---:|---:|---:|---:|"])
    for n in sorted({int(row["logical_qubits"]) for row in rows}):
        group = [row for row in rows if int(row["logical_qubits"]) == n
                 and row["routing_status"] == "routed"]
        if not group:
            lines.append(f"| {n} | 0 | n/a | n/a | n/a | n/a |")
            continue
        base = sum(int(row["stock_2q"]) for row in group)
        trial = sum(int(row["raw_2q"]) for row in group)
        wins = sum(int(row["raw_2q"]) < int(row["stock_2q"]) for row in group)
        ties = sum(int(row["raw_2q"]) == int(row["stock_2q"]) for row in group)
        lines.append(f"| {n} | {len(group)} | {base} | {trial} | "
                     f"{(trial / base - 1) * 100:+.1f}% | {wins}/{ties}/{len(group) - wins - ties} |")
    errors = [float(row["max_route_state_error"]) for row in rows
              if row["max_route_state_error"] not in ("", None)]
    mps = [float(row[column]) for row in rows
           for column in ("mps_stock_probability_error", "mps_raw_probability_error")
           if row[column] not in ("", None)]
    peaks = [float(row["peak_process_rss_mib"]) for row in rows
             if row["peak_process_rss_mib"] not in ("", None)]
    lines.extend(["", f"Symbolically checked routes: {sum(int(row['routes_verified']) for row in rows)}.",
                  f"Numerical route-state checks: {len(errors)} cases; max error "
                  f"{max(errors) if errors else 'n/a'}.",
                  f"Compiled MPS output checks: {len(mps)} circuits; max probability error "
                  f"{max(mps) if mps else 'n/a'}.",
                  f"Peak process RSS observed: {max(peaks):.1f} MiB." if peaks else
                  "Peak process RSS: unavailable on this platform.",
                  "Rows from three snapshots share logical workloads and are not fully independent.",
                  f"Full paired rows: {path.name}."])
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args, output: Path) -> None:
    rows = []
    simulator = None
    if args.mps16:
        from qiskit_aer import AerSimulator
        simulator = AerSimulator(method="matrix_product_state")
    for backend_name in args.backends:
        backend = BACKENDS[backend_name]()
        for n in args.sizes:
            for family in args.families:
                for offset in range(args.graphs):
                    graph_id = args.graph_start + offset
                    graph_seed = graph_id + 1_000_000 * n + 100_000 * list(FAMILIES).index(family)
                    edges = graph_edges(n, FAMILIES[family], graph_seed)
                    for layers in args.layers:
                        current, peak = process_memory_mib()
                        observed = max((value for value in (current, peak) if value is not None), default=0.0)
                        if observed > args.rss_stop_gib * 1024:
                            raise MemoryError(f"Process RSS {observed:.0f} MiB exceeds stop limit")
                        print(f"{backend.name} n={n} {family} graph={graph_id} p={layers}", flush=True)
                        qc = qaoa_circuit(n, edges, layers, graph_seed + 211 + layers)
                        start = perf_counter()
                        stock = v1.compile_stock_level3(qc.copy(), backend, seed_transpiler=args.seed)
                        stock_seconds = perf_counter() - start
                        stock_2q, stock_depth = native_metrics(stock, backend)
                        preferred = tuple(stock.circuit.layout.initial_index_layout()[:n])
                        start = perf_counter()
                        try:
                            variants, candidates = compare_variants(
                                qc.copy(), backend, seed_transpiler=args.seed,
                                layouts=args.layouts, finalists=args.finalists,
                                beam_width=args.beam_width,
                                beam_seconds=args.beam_seconds, preferred_layout=preferred)
                            raw = variants["tournament"]
                            status = "routed"
                        except LayoutSearchFailed:
                            raw, candidates = stock, []
                            status = "stock_fallback"
                        hybrid_seconds = perf_counter() - start + stock_seconds
                        state_errors = []
                        for candidate in candidates:
                            replay = route_replay(qc, candidate.routed, candidate.layout,
                                                  backend.coupling_map)
                            if n <= args.statevector_up_to:
                                logical_reference = QuantumCircuit(n, global_phase=qc.global_phase)
                                for instruction in qc.data:
                                    if isinstance(instruction.operation, PauliEvolutionGate):
                                        for (a, b), angle in sorted(v1._layer_terms(qc, instruction).items()):
                                            logical_reference.rzz(float(angle), a, b)
                                    else:
                                        logical_reference.append(instruction.operation,
                                                                 [qc.find_bit(instruction.qubits[0]).index])
                                error = state_error(logical_reference, replay)
                                if error > bench.STATE_TOL:
                                    raise AssertionError(f"Rust route changed logical state: {error}")
                                state_errors.append(error)
                        raw_2q, raw_depth = native_metrics(raw, backend)
                        guard = stock if (stock_2q, stock_depth) < (raw_2q, raw_depth) else raw
                        guard_2q, guard_depth = native_metrics(guard, backend)
                        mps_stock = mps_raw = ""
                        if simulator is not None and n == 16 and offset == 0 and layers == 1:
                            logical_reference = QuantumCircuit(n, global_phase=qc.global_phase)
                            for instruction in qc.data:
                                if isinstance(instruction.operation, PauliEvolutionGate):
                                    for (a, b), angle in sorted(v1._layer_terms(qc, instruction).items()):
                                        logical_reference.rzz(float(angle), a, b)
                                else:
                                    logical_reference.append(instruction.operation,
                                                             [qc.find_bit(instruction.qubits[0]).index])
                            ideal = np.abs(Statevector.from_instruction(logical_reference).data) ** 2
                            mps_stock = mps_probability_error(stock, ideal, simulator)
                            mps_raw = mps_probability_error(raw, ideal, simulator)
                        current, peak = process_memory_mib()
                        rows.append({
                            "snapshot": backend.name, "logical_qubits": n, "family": family,
                            "graph_id": graph_id, "layers": layers, "edges": len(edges),
                            "stock_2q": stock_2q, "raw_2q": raw_2q, "guarded_2q": guard_2q,
                            "stock_2q_depth": stock_depth, "raw_2q_depth": raw_depth,
                            "guarded_2q_depth": guard_depth,
                            "stock_seconds": stock_seconds, "raw_total_seconds": hybrid_seconds,
                            "guarded_total_seconds": hybrid_seconds, "routing_status": status,
                            "routes_verified": len(candidates),
                            "max_route_state_error": max(state_errors) if state_errors else "",
                            "mps_stock_probability_error": mps_stock,
                            "mps_raw_probability_error": mps_raw,
                            "peak_process_rss_mib": peak if peak is not None else "",
                        })
                        write_rows(output, rows)
                        write_report(output, rows, args)
                        observed = max((value for value in (current, peak) if value is not None), default=0.0)
                        if observed > args.rss_stop_gib * 1024:
                            raise MemoryError(f"Process RSS {observed:.0f} MiB exceeds stop limit")
    print(output.with_suffix(".md"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="16,20,24")
    parser.add_argument("--families", default="medium50")
    parser.add_argument("--graphs", type=int, default=1)
    parser.add_argument("--layers", default="1,2")
    parser.add_argument("--backends", default="FakeBrisbane,FakeKyiv,FakeSherbrooke")
    parser.add_argument("--graph-start", type=int, default=94000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--layouts", type=int, default=1)
    parser.add_argument("--finalists", type=int, default=1)
    parser.add_argument("--beam-seconds", type=float, default=1.0)
    parser.add_argument("--beam-width", type=int, default=v1.DEFAULT_BEAM_WIDTH)
    parser.add_argument("--statevector-up-to", type=int, default=20)
    parser.add_argument("--rss-stop-gib", type=float, default=12.0)
    parser.add_argument("--mps16", action="store_true")
    parser.add_argument("--output-name", default="v2_scaling_16_20_24")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        args.sizes = tuple(int(value) for value in args.sizes.split(","))
        args.layers = tuple(int(value) for value in args.layers.split(","))
    except ValueError:
        parser.error("--sizes and --layers need comma-separated integers")
    args.families = tuple(value.strip() for value in args.families.split(","))
    args.backends = tuple(value.strip() for value in args.backends.split(","))
    if (not args.sizes or any(n < 16 or n > 24 for n in args.sizes)
            or not args.layers or any(p not in (1, 2) for p in args.layers)):
        parser.error("Hybrid research supports 16..24 logical qubits; QAOA layers must be 1 or 2")
    if not args.families or any(f not in FAMILIES for f in args.families):
        parser.error("Unknown graph family")
    if not args.backends or any(name not in BACKENDS for name in args.backends):
        parser.error("Unknown fake backend")
    if (args.graphs < 1 or args.layouts < 1 or args.finalists < 1 or args.beam_width < 1
            or args.beam_seconds <= 0 or args.rss_stop_gib <= 0
            or args.statevector_up_to > 20 or args.statevector_up_to < 0):
        parser.error("Invalid graph/search/memory limit; exact state checks are capped at 20 qubits")
    if (not args.output_name.startswith("v2_scaling_")
            or Path(args.output_name).name != args.output_name):
        parser.error("--output-name must be a plain v2_scaling_* filename prefix")
    output = ROOT / "results" / f"{args.output_name}.csv"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Results already exist: {output}")
    run(args, output)


if __name__ == "__main__":
    main()
