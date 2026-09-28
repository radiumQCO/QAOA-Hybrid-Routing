"""Fresh paired holdout for Stock, dense Hybrid, and commuting + SAT."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKyiv, FakeSherbrooke

import benchmark as bench
import hybrid_level3 as stock_compiler
from hybrid_level3_v2 import compile_hybrid_dense_v2
from layout_seed import physical_line
from research_commuting_baseline import compile_commuting
from research_scaling import (
    graph_edges, mps_probability_error, native_metrics, process_memory_mib,
    qaoa_circuit, route_replay,
)
from rust_router_core import optimise_line_layout_rust, route_line_layers_rust


ROOT = Path(__file__).resolve().parent
BACKENDS = (FakeBrisbane, FakeKyiv, FakeSherbrooke)
FAMILIES_16 = {"medium50", "dense75", "er50", "er75"}
FIELDS = (
    "snapshot", "logical_qubits", "family", "graph_id", "graph_seed", "layers", "edges",
    "stock_2q", "hybrid_2q", "commuting_2q",
    "stock_2q_depth", "hybrid_2q_depth", "commuting_2q_depth",
    "stock_seconds", "hybrid_seconds", "commuting_seconds",
    "hybrid_status", "hybrid_mode", "line_symbolic_check",
    "mps_stock_error", "mps_hybrid_error", "mps_commuting_error", "peak_rss_mib",
)


def workloads(graph_start: int, count: int):
    """Use the same graph and angle generators as the earlier studies, with new IDs."""
    for spec in bench.graph_specs("standard", graph_start):
        if spec.family in FAMILIES_16 and spec.graph_id < graph_start + count:
            for layers in (1, 2):
                qc, _ = bench.qaoa_circuit(
                    spec.edges, layers, spec.graph_seed + 211 + layers)
                yield 16, spec.family, spec.graph_id, spec.graph_seed, layers, spec.edges, qc
    for n in (20, 24):
        for family, density in (("medium50", 0.50), ("dense75", 0.75)):
            for graph_id in range(graph_start, graph_start + count):
                seed = graph_id + 1_000_000 * n + (100_000 if family == "dense75" else 0)
                edges = graph_edges(n, density, seed)
                for layers in (1, 2):
                    qc = qaoa_circuit(n, edges, layers, seed + 211 + layers)
                    yield n, family, graph_id, seed, layers, edges, qc


def key(row) -> tuple[str, int, str, int, int]:
    return (row["snapshot"], int(row["logical_qubits"]), row["family"],
            int(row["graph_id"]), int(row["layers"]))


def assert_fresh(graph_start: int, count: int, output: Path) -> None:
    selected = set(range(graph_start, graph_start + count))
    for path in (ROOT / "results").glob("*.csv"):
        if path.resolve() == output.resolve():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "graph_id" not in (reader.fieldnames or []):
                continue
            for row in reader:
                try:
                    graph_id = int(row["graph_id"])
                except (TypeError, ValueError):
                    continue
                if graph_id in selected:
                    raise ValueError(f"Graph ID {graph_id} already appears in {path.name}")


def ideal_probabilities(qc: QuantumCircuit) -> np.ndarray:
    """Use only a 16-qubit statevector for the selected MPS checks."""
    if qc.num_qubits != 16:
        raise ValueError("Full statevector checks are limited to 16 logical qubits")
    logical = QuantumCircuit(16, global_phase=qc.global_phase)
    for inst in qc.data:
        if isinstance(inst.operation, PauliEvolutionGate):
            for (u, v), angle in sorted(stock_compiler._layer_terms(qc, inst).items()):
                logical.rzz(float(angle), u, v)
        else:
            logical.append(inst.operation, [qc.find_bit(inst.qubits[0]).index])
    return np.abs(Statevector.from_instruction(logical).data) ** 2


def save(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def report(path: Path, rows: list[dict], graph_start: int, count: int) -> None:
    lines = [
        "# Final fresh MaxCut holdout", "",
        f"Graph IDs {graph_start}–{graph_start + count - 1}; compiler seed 11; "
        "one and two QAOA layers. Full FakeBrisbane, FakeKyiv, and FakeSherbrooke targets. "
        "Stock Level 3, raw dense Hybrid, and Qiskit commuting + SAT were compiled "
        "in the same run on paired circuits. Hybrid uses a 0.15 s Rust line-order "
        "search and up to 1 s of Rust beam search. SAT allows 2 s per feasibility "
        "check. This is a local compilation benchmark, not a QPU run.", "",
        "The same logical circuits are repeated on three targets. A Hybrid fallback "
        "to Stock is counted as a routing failure. Lower native 2Q is better.", "",
        "| Logical size | Cases | Hybrid routes | Stock 2Q | Hybrid 2Q | "
        "Commuting 2Q | Hybrid vs commuting | Hybrid W/T/L vs commuting | "
        "Hybrid W/T/L vs Stock |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in (16, 20, 24):
        group = [r for r in rows if int(r["logical_qubits"]) == n]
        if not group:
            continue
        stock = sum(int(r["stock_2q"]) for r in group)
        hybrid = sum(int(r["hybrid_2q"]) for r in group)
        commuting = sum(int(r["commuting_2q"]) for r in group)
        against_c = [sum(int(r["hybrid_2q"]) < int(r["commuting_2q"]) for r in group),
                     sum(int(r["hybrid_2q"]) == int(r["commuting_2q"]) for r in group)]
        against_s = [sum(int(r["hybrid_2q"]) < int(r["stock_2q"]) for r in group),
                     sum(int(r["hybrid_2q"]) == int(r["stock_2q"]) for r in group)]
        lines.append(f"| {n} | {len(group)} | "
                     f"{sum(r['hybrid_status'] == 'routed' for r in group)} | "
                     f"{stock} | {hybrid} | {commuting} | "
                     f"{(hybrid / commuting - 1) * 100:+.1f}% | "
                     f"{against_c[0]}/{against_c[1]}/{len(group) - sum(against_c)} | "
                     f"{against_s[0]}/{against_s[1]}/{len(group) - sum(against_s)} |")
    lines += ["", "## By graph family", "",
              "| Size | Family | Cases | Hybrid routes | Stock 2Q | Hybrid 2Q | "
              "Commuting 2Q | Hybrid vs commuting |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for n in (16, 20, 24):
        for family in ("medium50", "er50", "dense75", "er75"):
            group = [r for r in rows if int(r["logical_qubits"]) == n and r["family"] == family]
            if not group:
                continue
            h = sum(int(r["hybrid_2q"]) for r in group)
            c = sum(int(r["commuting_2q"]) for r in group)
            lines.append(f"| {n} | {family} | {len(group)} | "
                         f"{sum(r['hybrid_status'] == 'routed' for r in group)} | "
                         f"{sum(int(r['stock_2q']) for r in group)} | {h} | {c} | "
                         f"{(h / c - 1) * 100:+.1f}% |")
    lines += ["", "## Time, depth, and checks", "",
              "| Size | Median Stock s | Median Hybrid s | Median commuting s | "
              "Stock 2Q depth | Hybrid 2Q depth | Commuting 2Q depth |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for n in (16, 20, 24):
        group = [r for r in rows if int(r["logical_qubits"]) == n]
        if not group:
            continue
        lines.append(f"| {n} | "
                     f"{median(float(r['stock_seconds']) for r in group):.3f} | "
                     f"{median(float(r['hybrid_seconds']) for r in group):.3f} | "
                     f"{median(float(r['commuting_seconds']) for r in group):.3f} | "
                     f"{sum(int(r['stock_2q_depth']) for r in group)} | "
                     f"{sum(int(r['hybrid_2q_depth']) for r in group)} | "
                     f"{sum(int(r['commuting_2q_depth']) for r in group)} |")
    symbolic = sum(r["line_symbolic_check"] == "passed" for r in rows)
    errors = [float(r[field]) for r in rows
              for field in ("mps_stock_error", "mps_hybrid_error", "mps_commuting_error")
              if r[field] not in ("", None)]
    peak = [float(r["peak_rss_mib"]) for r in rows if r["peak_rss_mib"] not in ("", None)]
    lines += ["", f"Dense line-route symbolic checks: {symbolic}. "
              "This verifies a dense line candidate, which may differ from the final "
              "tournament choice. Native target support is checked for all three "
              "compiled circuits on every completed row.",
              f"Selected 16q compiled MPS probability checks: {len(errors)}; "
              f"maximum error {max(errors) if errors else 'n/a'}. "
              "No full statevector is created above 16 logical qubits.",
              f"Peak process RSS: {max(peak):.1f} MiB." if peak else "Peak RSS unavailable.",
              f"Per-case data: {path.name}."]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-start", type=int, default=98000)
    parser.add_argument("--graphs-per-family", type=int, default=5)
    parser.add_argument("--output-name", default="v2_final_holdout_20260928")
    parser.add_argument("--mps16", action="store_true",
                        help="MPS-check 4 selected 16q cases, all three compilers")
    parser.add_argument("--plan-only", action="store_true",
                        help="show counts and check fresh graph IDs without compiling")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.graphs_per_family <= 12 or args.graph_start < 0 \
            or args.output_name != Path(args.output_name).name \
            or not args.output_name.startswith("v2_"):
        parser.error("Use a positive graph count and a plain v2_* output name")
    output = ROOT / "results" / f"{args.output_name}.csv"
    if output.exists() and not args.resume and not args.plan_only:
        raise FileExistsError(f"Would overwrite existing results: {output}")
    assert_fresh(args.graph_start, args.graphs_per_family, output)
    specs = list(workloads(args.graph_start, args.graphs_per_family))
    expected = len(specs) * len(BACKENDS)
    if args.plan_only:
        print(f"Graph IDs {args.graph_start}–{args.graph_start + args.graphs_per_family - 1} "
              f"are unused in existing result CSVs.")
        print(f"{len(specs)} distinct logical circuits × {len(BACKENDS)} targets "
              f"= {expected} paired compiler cases. Sizes: 16, 20, 24. "
              "No benchmark was run.")
        return
    rows = (list(csv.DictReader(output.open(newline="", encoding="utf-8")))
            if output.exists() else [])
    seen = {key(row) for row in rows}
    if len(seen) != len(rows):
        raise AssertionError("Duplicate rows in existing output")
    simulator = AerSimulator(method="matrix_product_state") if args.mps16 else None
    paths = {}
    for backend_class in BACKENDS:
        backend = backend_class()
        paths[backend.name] = physical_line(backend.coupling_map, 24)
        for n, family, graph_id, graph_seed, layers, edges, qc in specs:
            identity = (backend.name, n, family, graph_id, layers)
            if identity in seen:
                continue
            print(f"[{len(rows) + 1}/{expected}] {backend.name} {n}q {family} "
                  f"graph={graph_id} p={layers}", flush=True)
            started = perf_counter()
            stock = stock_compiler.compile_stock_level3(
                qc.copy(), backend, seed_transpiler=11)
            stock_seconds = perf_counter() - started
            started = perf_counter()
            hybrid = compile_hybrid_dense_v2(qc.copy(), backend, seed_transpiler=11)
            hybrid_seconds = perf_counter() - started
            started = perf_counter()
            first_layer = next(inst for inst in qc.data
                               if isinstance(inst.operation, PauliEvolutionGate))
            pairs = tuple(sorted(stock_compiler._layer_terms(qc, first_layer)))
            commuting, commuting_2q, commuting_depth, _, _, _, _, _ = compile_commuting(
                qc, backend, pairs, paths[backend.name][:n], 11, 2.0)
            commuting_seconds = perf_counter() - started
            stock_2q, stock_depth = native_metrics(stock, backend)
            hybrid_2q, hybrid_depth = native_metrics(hybrid, backend)
            symbolic = ""
            if family in ("dense75", "er75"):
                path = paths[backend.name][:n]
                order = optimise_line_layout_rust(qc, seed=11, max_seconds=0.15)
                layout = tuple(path[site] for site in order)
                routed = route_line_layers_rust(qc.copy(), backend.coupling_map, path, layout)
                route_replay(qc, routed, layout, backend.coupling_map)
                symbolic = "passed"
            row = dict(
                snapshot=backend.name, logical_qubits=n, family=family,
                graph_id=graph_id, graph_seed=graph_seed,
                layers=layers, edges=len(edges),
                stock_2q=stock_2q, hybrid_2q=hybrid_2q, commuting_2q=commuting_2q,
                stock_2q_depth=stock_depth, hybrid_2q_depth=hybrid_depth,
                commuting_2q_depth=commuting_depth,
                stock_seconds=stock_seconds, hybrid_seconds=hybrid_seconds,
                commuting_seconds=commuting_seconds,
                hybrid_status=("routed" if hybrid.route_mode.startswith("v2_rust")
                               else "stock_fallback"),
                hybrid_mode=hybrid.route_mode, line_symbolic_check=symbolic,
                mps_stock_error="", mps_hybrid_error="", mps_commuting_error="",
                peak_rss_mib="",
            )
            if simulator is not None and backend.name == "fake_brisbane" and n == 16 \
                    and graph_id == args.graph_start and family in ("medium50", "dense75"):
                ideal = ideal_probabilities(qc)
                for name, compiled in (("stock", stock), ("hybrid", hybrid),
                                       ("commuting", commuting)):
                    row[f"mps_{name}_error"] = mps_probability_error(
                        compiled, ideal, simulator)
            _, peak = process_memory_mib()
            row["peak_rss_mib"] = peak if peak is not None else ""
            rows.append(row)
            seen.add(identity)
            save(output, rows)
            report(output, rows, args.graph_start, args.graphs_per_family)
    if len(rows) != expected:
        raise AssertionError(f"Expected {expected} rows, found {len(rows)}")
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
