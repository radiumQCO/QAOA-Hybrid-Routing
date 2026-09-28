"""Compare final Qiskit compilation with and without rerouting the same Rust route."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKyiv, FakeSherbrooke

import benchmark as bench
import hybrid_level3 as base
from hybrid_level3_v2 import _error_cost
from research_ibm_full import selected_specs
from research_scaling import (
    graph_edges, mps_probability_error, native_metrics, process_memory_mib,
    qaoa_circuit, route_replay,
)
from rust_router_core import route_beam_layers_rust


ROOT = Path(__file__).resolve().parent
BACKENDS = (FakeBrisbane, FakeKyiv, FakeSherbrooke)
FIELDS = (
    "suite", "snapshot", "logical_qubits", "family", "graph_id", "layers",
    "edges", "route_status", "rust_swaps", "stock_2q", "sabre_2q", "none_2q",
    "stock_2q_depth", "sabre_2q_depth", "none_2q_depth",
    "stock_depth", "sabre_depth", "none_depth",
    "stock_error", "sabre_error", "none_error",
    "stock_seconds", "route_seconds", "sabre_final_seconds", "none_final_seconds",
    "mps_stock_error", "mps_sabre_error", "mps_none_error", "peak_rss_mib",
)


def cases(args):
    for backend_class in BACKENDS:
        backend = backend_class()
        if args.suite in ("all", "full16"):
            for spec in selected_specs(86000, 10):
                for layers in (1, 2):
                    qc, _ = bench.qaoa_circuit(
                        spec.edges, layers, spec.graph_seed + 211 + layers)
                    yield "full16", backend, qc, spec.family, spec.graph_id, layers, len(spec.edges)
        if args.suite in ("all", "scaling"):
            for n in (16, 20, 24):
                for family, density in (("medium50", 0.50), ("dense75", 0.75)):
                    for graph_id in range(94000, 94003):
                        seed = graph_id + 1_000_000 * n + (100_000 if family == "dense75" else 0)
                        edges = graph_edges(n, density, seed)
                        for layers in (1, 2):
                            qc = qaoa_circuit(n, edges, layers, seed + 211 + layers)
                            yield "scaling", backend, qc, family, graph_id, layers, len(edges)


def final_compile(routed, backend, method: str, seed: int, density: float):
    start = perf_counter()
    manager = generate_preset_pass_manager(
        optimization_level=3,
        target=backend.target,
        initial_layout=list(range(backend.num_qubits)),
        routing_method=method,
        seed_transpiler=seed,
        approximation_degree=1.0,
        qubits_initially_zero=False,
        scheduling_method="alap",
    )
    circuit = manager.run(routed.circuit)
    elapsed = perf_counter() - start
    after = base._final_positions(circuit, backend.num_qubits)
    positions = [after[q] for q in routed.final_positions]
    result = base.CompileResult(circuit, positions, f"v2_rust_{method}", method,
                                density, 0.0, elapsed, routed.swaps)
    gates, depth = native_metrics(result, backend)
    return result, gates, depth, elapsed


def save(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summary_line(label: str, group: list[dict]) -> str:
    routed = sum(row["route_status"] == "routed" for row in group)
    stock = sum(int(row["stock_2q"]) for row in group)
    sabre = sum(int(row["sabre_2q"]) for row in group)
    none = sum(int(row["none_2q"]) for row in group)
    wins = sum(int(row["none_2q"]) < int(row["stock_2q"]) for row in group)
    ties = sum(int(row["none_2q"]) == int(row["stock_2q"]) for row in group)
    losses = len(group) - wins - ties
    stock_time = median(float(row["stock_seconds"]) for row in group)
    none_time = median(sum(float(row[key]) for key in (
        "stock_seconds", "route_seconds", "none_final_seconds")) for row in group)
    return (f"| {label} | {len(group)} | {routed} | {stock} | {sabre} | {none} | "
            f"{(none / stock - 1) * 100:+.1f}% | {wins}/{ties}/{losses} | "
            f"{stock_time:.3f} | {none_time:.3f} |")


def report(path: Path, rows: list[dict], args) -> None:
    lines = [
        "# Rerouting an already routed QAOA circuit", "",
        "Stock Qiskit Level 3 uses SABRE. For each successful Hybrid case, one Rust route "
        "is compiled twice with identical Qiskit settings except for the final routing method "
        "(`sabre` or `none`). Stock, both Hybrid outputs, and the Rust route use the same "
        "logical workload, target, and Qiskit seed 11. A Rust timeout is recorded as a "
        "Stock fallback in both Hybrid columns.",
        "These are local compilation results on full 127-qubit fake targets, not QPU measurements. "
        "All reported Hybrid times include the Stock compile used to obtain its starting layout.",
        "The 240-row full16 suite repeats 80 graph/layer workloads across three snapshots; "
        "the 108-row scaling suite repeats 36 workloads. Snapshot rows are not independent graphs.",
        "All successful Rust routes are checked by exact symbolic gate/SWAP replay. Both compiled "
        "outputs are checked for supported native 2Q gates and valid final logical positions. "
        "Selected 16-qubit outputs are also checked by MPS probabilities, which do not prove phase equivalence.",
        "", "| Suite / size | Cases | Rust routes | Stock 2Q | Hybrid + SABRE 2Q | Hybrid + none 2Q | "
        "None vs Stock | Wins/ties/losses | Median Stock s | Median Hybrid + none s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for suite, n in (("full16", 16), ("scaling", 16), ("scaling", 20), ("scaling", 24)):
        group = [row for row in rows if row["suite"] == suite and int(row["logical_qubits"]) == n]
        if group:
            lines.append(summary_line(f"{suite} {n}q", group))
    lines += ["", "## Depth and guarded compiler", "",
              "The guarded compiler compiles both Stock and Hybrid, then selects the lower "
              "native 2Q count (2Q depth breaks ties). It is a portfolio result, not a "
              "standalone Rust-router result.", "",
              "| Suite / size | Mean Stock 2Q depth | Mean raw Hybrid 2Q depth | "
              "Guarded 2Q | Guarded vs Stock | Stock chosen |",
              "|---|---:|---:|---:|---:|---:|"]
    for suite, n in (("full16", 16), ("scaling", 16), ("scaling", 20), ("scaling", 24)):
        group = [row for row in rows if row["suite"] == suite and int(row["logical_qubits"]) == n]
        if not group:
            continue
        stock = sum(int(row["stock_2q"]) for row in group)
        guarded = sum(min(int(row["stock_2q"]), int(row["none_2q"])) for row in group)
        stock_chosen = sum((int(row["stock_2q"]), int(row["stock_2q_depth"])) <
                           (int(row["none_2q"]), int(row["none_2q_depth"])) for row in group)
        stock_depth = sum(int(row["stock_2q_depth"]) for row in group) / len(group)
        none_depth = sum(int(row["none_2q_depth"]) for row in group) / len(group)
        lines.append(f"| {suite} {n}q | {stock_depth:.1f} | {none_depth:.1f} | "
                     f"{guarded} | {(guarded / stock - 1) * 100:+.1f}% | {stock_chosen} |")
    lines += ["", "## Successful Rust routes only", "",
              "This table removes Stock fallback rows from all three columns.", "",
              "| Suite / size | Routes | Stock 2Q | Hybrid + SABRE 2Q | Hybrid + none 2Q | "
              "None vs Stock | None vs SABRE |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for suite, n in (("full16", 16), ("scaling", 16), ("scaling", 20), ("scaling", 24)):
        group = [row for row in rows if row["suite"] == suite and int(row["logical_qubits"]) == n
                 and row["route_status"] == "routed"]
        if not group:
            continue
        stock = sum(int(row["stock_2q"]) for row in group)
        sabre = sum(int(row["sabre_2q"]) for row in group)
        none = sum(int(row["none_2q"]) for row in group)
        lines.append(f"| {suite} {n}q | {len(group)} | {stock} | {sabre} | {none} | "
                     f"{(none / stock - 1) * 100:+.1f}% | {(none / sabre - 1) * 100:+.1f}% |")
    lines += ["", "## By graph family", "",
              "| Suite / size | Family | Cases | Routes | None vs Stock 2Q |",
              "|---|---|---:|---:|---:|"]
    for suite, n in (("full16", 16), ("scaling", 16), ("scaling", 20), ("scaling", 24)):
        for family in ("medium50", "er50", "dense75", "er75"):
            group = [row for row in rows if row["suite"] == suite
                     and int(row["logical_qubits"]) == n and row["family"] == family]
            if group:
                stock = sum(int(row["stock_2q"]) for row in group)
                none = sum(int(row["none_2q"]) for row in group)
                routed = sum(row["route_status"] == "routed" for row in group)
                lines.append(f"| {suite} {n}q | {family} | {len(group)} | {routed} | "
                             f"{(none / stock - 1) * 100:+.1f}% |")
    lines += ["", "## Saved calibration error estimate", "",
              "For successful routes with complete calibration data, lower is better. "
              "This score is only a sum of saved gate-error estimates; it is not measured QPU fidelity.", "",
              "| Suite / size | Paired cases | Hybrid lower/tie/higher |",
              "|---|---:|---:|"]
    for suite, n in (("full16", 16), ("scaling", 16), ("scaling", 20), ("scaling", 24)):
        pairs = [(float(row["stock_error"]), float(row["none_error"])) for row in rows
                 if row["suite"] == suite and int(row["logical_qubits"]) == n
                 and row["route_status"] == "routed"
                 and row["stock_error"] not in ("", None)
                 and row["none_error"] not in ("", None)]
        if pairs:
            wins = sum(new < old - 1e-12 for old, new in pairs)
            ties = sum(abs(new - old) <= 1e-12 for old, new in pairs)
            losses = len(pairs) - wins - ties
            lines.append(f"| {suite} {n}q | {len(pairs)} | {wins}/{ties}/{losses} |")
    mps = [float(row[key]) for row in rows for key in
           ("mps_stock_error", "mps_sabre_error", "mps_none_error") if row[key] not in ("", None)]
    rss = [float(row["peak_rss_mib"]) for row in rows if row["peak_rss_mib"] not in ("", None)]
    lines += ["", f"Rows: {len(rows)}; successful Rust routes: "
              f"{sum(row['route_status'] == 'routed' for row in rows)}.",
              f"Selected compiled MPS probability checks: {len(mps)}; maximum error "
              f"{max(mps) if mps else 'n/a'}.",
              f"Peak process RSS: {max(rss):.1f} MiB." if rss else "Peak process RSS: n/a.",
              f"Per-case data: {path.name}."]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args, path: Path) -> None:
    rows = []
    simulator = AerSimulator(method="matrix_product_state") if args.mps16 else None
    total = (240 if args.suite in ("all", "full16") else 0) + (
        108 if args.suite in ("all", "scaling") else 0)
    for index, (suite, backend, qc, family, graph_id, layers, edges) in enumerate(cases(args), 1):
        current, peak = process_memory_mib()
        if max((value for value in (current, peak) if value is not None), default=0.0) > args.rss_stop_gib * 1024:
            raise MemoryError("Process memory exceeded the configured limit")
        n = qc.num_qubits
        print(f"[{index}/{total}] {suite} {backend.name} n={n} "
              f"{family} graph={graph_id} p={layers}", flush=True)
        start = perf_counter()
        stock = base.compile_stock_level3(qc.copy(), backend, seed_transpiler=11)
        stock_seconds = perf_counter() - start
        stock_2q, stock_depth = native_metrics(stock, backend)
        layout = tuple(stock.circuit.layout.initial_index_layout()[:n])
        start = perf_counter()
        try:
            routed = route_beam_layers_rust(qc.copy(), backend.coupling_map, list(layout),
                                            max_seconds=args.beam_seconds)
        except (base.BeamBudgetExceeded, base.BeamRoutingFailed) as exc:
            routed = None
            status = type(exc).__name__
        else:
            status = "routed"
            replay = route_replay(qc, routed, layout, backend.coupling_map)
        route_seconds = perf_counter() - start
        results = {}
        if routed is not None:
            density = base.inspect_qaoa_shape(qc).density
            order = ("sabre", "none") if index % 2 else ("none", "sabre")
            for method in order:
                results[method] = final_compile(routed, backend, method, 11, density)
        row = {
            "suite": suite, "snapshot": backend.name, "logical_qubits": n,
            "family": family, "graph_id": graph_id, "layers": layers, "edges": edges,
            "route_status": status, "rust_swaps": routed.swaps if routed else "",
            "stock_2q": stock_2q, "sabre_2q": results["sabre"][1] if results else stock_2q,
            "none_2q": results["none"][1] if results else stock_2q,
            "stock_2q_depth": stock_depth,
            "sabre_2q_depth": results["sabre"][2] if results else stock_depth,
            "none_2q_depth": results["none"][2] if results else stock_depth,
            "stock_depth": stock.circuit.depth(),
            "sabre_depth": results["sabre"][0].circuit.depth() if results else stock.circuit.depth(),
            "none_depth": results["none"][0].circuit.depth() if results else stock.circuit.depth(),
            "stock_error": _error_cost(stock.circuit, backend.target),
            "sabre_error": _error_cost(results["sabre"][0].circuit, backend.target) if results else "",
            "none_error": _error_cost(results["none"][0].circuit, backend.target) if results else "",
            "stock_seconds": stock_seconds, "route_seconds": route_seconds,
            "sabre_final_seconds": results["sabre"][3] if results else 0.0,
            "none_final_seconds": results["none"][3] if results else 0.0,
            "mps_stock_error": "", "mps_sabre_error": "", "mps_none_error": "",
        }
        if simulator is not None and suite == "full16" and backend.name == "fake_brisbane" \
                and family in ("medium50", "dense75") and graph_id == 86000 and layers == 1 \
                and routed is not None:
            ideal = np.abs(Statevector.from_instruction(replay).data) ** 2
            for name, compiled in (("stock", stock), ("sabre", results["sabre"][0]),
                                   ("none", results["none"][0])):
                row[f"mps_{name}_error"] = mps_probability_error(compiled, ideal, simulator)
        current, peak = process_memory_mib()
        row["peak_rss_mib"] = peak if peak is not None else ""
        rows.append(row)
        save(path, rows)
    report(path, rows, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "full16", "scaling"), default="all")
    parser.add_argument("--beam-seconds", type=float, default=1.0)
    parser.add_argument("--rss-stop-gib", type=float, default=12.0)
    parser.add_argument("--mps16", action="store_true")
    parser.add_argument("--output-name", default="v2_no_reroute_ablation")
    args = parser.parse_args()
    if args.beam_seconds <= 0 or args.rss_stop_gib <= 0:
        parser.error("Search and memory limits must be positive")
    if args.output_name != Path(args.output_name).name or not args.output_name.startswith("v2_"):
        parser.error("--output-name must be a plain v2_* filename prefix")
    path = ROOT / "results" / f"{args.output_name}.csv"
    if path.exists():
        raise FileExistsError(f"Would overwrite existing results: {path}")
    run(args, path)


if __name__ == "__main__":
    main()
