"""Paired 64q test of plain line routing against optimized line ordering."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from time import perf_counter

from hybrid_level3 import _final_positions, inspect_qaoa_shape
from hybrid_level3_v2 import Candidate, _compile_candidate
from layout_seed import physical_line
from research_32plus import BACKENDS, DENSITIES
from research_final_holdout import assert_fresh
from research_scaling import graph_edges, native_metrics, process_memory_mib, qaoa_circuit, route_replay
from rust_router_core import optimise_line_layout_rust, route_line_layers_rust


ROOT = Path(__file__).resolve().parent
FIELDS = (
    "snapshot", "family", "graph_id", "layers", "edges",
    "plain_2q", "optimized_2q", "plain_depth", "optimized_depth",
    "plain_seconds", "optimized_seconds", "ordering_seconds", "peak_rss_mib",
)


def compile_variant(qc, backend, path, optimized: bool) -> dict:
    started = perf_counter()
    ordering_seconds = 0.0
    if optimized:
        order_started = perf_counter()
        positions = optimise_line_layout_rust(qc, seed=11, max_seconds=0.15)
        ordering_seconds = perf_counter() - order_started
        layout = tuple(path[position] for position in positions)
    else:
        layout = path

    routed = route_line_layers_rust(qc.copy(), backend.coupling_map, path, layout)
    route_replay(qc, routed, layout, backend.coupling_map)
    candidate = Candidate(layout, routed, 0.0)
    _compile_candidate(candidate, backend, 11, inspect_qaoa_shape(qc).density)
    seconds = perf_counter() - started
    count, depth = native_metrics(candidate.compiled, backend)
    output_positions = _final_positions(candidate.compiled.circuit, backend.num_qubits)
    expected = tuple(output_positions[q] for q in routed.final_positions)
    if expected != tuple(candidate.compiled.final_positions):
        raise AssertionError("Compiled final mapping differs from the symbolic route")
    return {"2q": count, "depth": depth, "seconds": seconds,
            "ordering_seconds": ordering_seconds}


def save(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def report(rows: list[dict], first_graph: int, graphs: int) -> str:
    lines = ["# 64q line-ordering ablation", "",
             f"Graph IDs {first_graph}–{first_graph + graphs - 1}. The two variants "
             "use identical circuits, physical lines, Qiskit seed, and native compiler. "
             "The optimized variant includes the 0.15-second ordering search in its time. "
             "Each variant builds exactly one line route; there is no tournament or beam. "
             "All recorded routes pass symbolic SWAP/gate replay, final-mapping, coupling, "
             "and target-native 2Q checks. No full statevector or QPU execution is used.", "",
             "| Family | Cases | Plain 2Q | Optimized 2Q | Change | "
             "Plain depth | Optimized depth | W/T/L | Mean plain / optimized time |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for family, group in (("All", rows),) + tuple(
            (family, [row for row in rows if row["family"] == family])
            for family in DENSITIES):
        if not group:
            continue
        plain = sum(int(row["plain_2q"]) for row in group)
        optimized = sum(int(row["optimized_2q"]) for row in group)
        wins = sum(int(row["optimized_2q"]) < int(row["plain_2q"]) for row in group)
        ties = sum(int(row["optimized_2q"]) == int(row["plain_2q"]) for row in group)
        lines.append(
            f"| {family} | {len(group)} | {plain} | {optimized} | "
            f"{100 * (optimized / plain - 1):+.1f}% | "
            f"{sum(int(row['plain_depth']) for row in group)} | "
            f"{sum(int(row['optimized_depth']) for row in group)} | "
            f"{wins}/{ties}/{len(group) - wins - ties} | "
            f"{mean(float(row['plain_seconds']) for row in group):.2f} / "
            f"{mean(float(row['optimized_seconds']) for row in group):.2f} s |"
        )
    if rows:
        lines += ["", f"Peak process RAM: "
                  f"{max(float(row['peak_rss_mib']) for row in rows):.0f} MiB."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-start", type=int, default=101100)
    parser.add_argument("--graphs", type=int, default=5)
    parser.add_argument("--rss-stop-gib", type=float, default=8.0)
    parser.add_argument("--output-name", default="v2_64q_line_order_ablation_20260928")
    args = parser.parse_args()
    if (args.graphs < 1 or args.rss_stop_gib <= 0
            or not args.output_name.replace("_", "").isalnum()):
        parser.error("Invalid graph count, RAM limit, or output name")
    output = ROOT / "results" / f"{args.output_name}.csv"
    report_path = output.with_suffix(".md")
    assert_fresh(args.graph_start, args.graphs, output)
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8"))) if output.exists() else []
    seen = {(row["snapshot"], row["family"], int(row["graph_id"]), int(row["layers"]))
            for row in rows}
    if len(seen) != len(rows):
        raise AssertionError("Duplicate ablation rows")
    total = len(BACKENDS) * len(DENSITIES) * args.graphs * 2
    for backend_name, backend_cls in BACKENDS.items():
        backend = backend_cls()
        path = physical_line(backend.coupling_map, 64)
        for family, density in DENSITIES.items():
            for graph_id in range(args.graph_start, args.graph_start + args.graphs):
                seed = graph_id + 64_000_000 + (100_000 if family == "dense75" else 0)
                edges = graph_edges(64, density, seed)
                for layers in (1, 2):
                    key = (backend.name, family, graph_id, layers)
                    if key in seen:
                        continue
                    _, peak = process_memory_mib()
                    if peak is not None and peak > args.rss_stop_gib * 1024:
                        raise MemoryError(f"Stopping at {peak:.0f} MiB peak RAM")
                    print(f"[{len(rows) + 1}/{total}] {backend.name} {family} "
                          f"graph={graph_id} p={layers}", flush=True)
                    qc = qaoa_circuit(64, edges, layers, seed + 211 + layers)
                    order = ("plain", "optimized") if len(rows) % 2 == 0 else ("optimized", "plain")
                    variants = {name: compile_variant(qc, backend, path, name == "optimized")
                                for name in order}
                    _, peak = process_memory_mib()
                    row = {"snapshot": backend.name, "family": family,
                           "graph_id": graph_id, "layers": layers, "edges": len(edges),
                           "peak_rss_mib": peak,
                           "ordering_seconds": variants["optimized"]["ordering_seconds"]}
                    for name, value in variants.items():
                        for metric in ("2q", "depth", "seconds"):
                            row[f"{name}_{metric}"] = value[metric]
                    rows.append(row)
                    seen.add(key)
                    save(output, rows)
                    report_path.write_text(report(rows, args.graph_start, args.graphs), encoding="utf-8")
    print(report(rows, args.graph_start, args.graphs), flush=True)


if __name__ == "__main__":
    main()
