"""Fresh paired check of frozen v2.0.0 and the v2.0.1 line routing path."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from time import perf_counter

from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKyiv, FakeSherbrooke

import hybrid_level3 as base
from hybrid_level3_v2 import compile_hybrid_dense_v2
from hybrid_line_next import compile_hybrid_line_next
from research_final_holdout import assert_fresh
from research_scaling import graph_edges, native_metrics, process_memory_mib, qaoa_circuit, route_replay
from rust_router_core import optimise_line_layout_rust


ROOT = Path(__file__).resolve().parent
BACKENDS = {"fake_brisbane": FakeBrisbane, "fake_kyiv": FakeKyiv,
            "fake_sherbrooke": FakeSherbrooke}
FAMILIES = {"medium50": 0.50, "dense75": 0.75}
FIELDS = ("snapshot", "logical_qubits", "family", "graph_id", "layers",
          "frozen_2q", "next_2q", "frozen_depth", "next_depth",
          "frozen_seconds", "next_seconds", "next_route_mode",
          "route_check", "mapping_check", "peak_rss_mib")


def report(rows, args):
    lines = ["# v2.0.1 line router: fresh paired confirmation", "",
             f"Graph IDs {args.graph_start}–{args.graph_start + args.graphs - 1} were not "
             "used for development. Frozen v2.0.0 compiles plain and optimized "
             "line candidates. The v2.0.1 path compiles one optimized line "
             "and, on medium graphs, ranks three 0.15-second Rust orders by "
             "symbolic SWAP count. Both use the same IBM fake target, native "
             "compiler, Qiskit seed, input circuit, and exactly the same seed-11 "
             "line order for each paired case. Timings include order "
             "search and native compilation, with verification outside the timer. "
             "The new routes pass symbolic "
             "SWAP/ZZ replay, final mapping, coupling, and native 2Q checks. "
             "No full 32/64q statevector or QPU run.", "",
             "| Size / family | Cases | Next wins / ties / losses | Frozen 2Q | Next 2Q | "
             "2Q change | Frozen depth | Next depth | Mean frozen / next time |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n, family in sorted({(int(r["logical_qubits"]), r["family"]) for r in rows}):
        group = [r for r in rows if int(r["logical_qubits"]) == n and r["family"] == family]
        f = sum(int(r["frozen_2q"]) for r in group)
        x = sum(int(r["next_2q"]) for r in group)
        wins = sum(int(r["next_2q"]) < int(r["frozen_2q"]) for r in group)
        ties = sum(int(r["next_2q"]) == int(r["frozen_2q"]) for r in group)
        lines.append(f"| {n}q {family} | {len(group)} | "
                     f"{wins}/{ties}/{len(group)-wins-ties} | {f} | {x} | "
                     f"{100 * (x / f - 1):+.1f}% | "
                     f"{sum(int(r['frozen_depth']) for r in group)} | "
                     f"{sum(int(r['next_depth']) for r in group)} | "
                     f"{mean(float(r['frozen_seconds']) for r in group):.2f} / "
                     f"{mean(float(r['next_seconds']) for r in group):.2f} s |")
    if rows:
        lines += ["", f"Route and compiled mapping checks: "
                  f"{sum(r['route_check'] == 'passed' for r in rows)}/{len(rows)} and "
                  f"{sum(r['mapping_check'] == 'passed' for r in rows)}/{len(rows)}. "
                  f"Peak process RAM: {max(float(r['peak_rss_mib']) for r in rows):.0f} MiB."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="32,64")
    parser.add_argument("--families", default="medium50,dense75")
    parser.add_argument("--layers", default="1,2")
    parser.add_argument("--backends", default="fake_brisbane,fake_kyiv,fake_sherbrooke")
    parser.add_argument("--graph-start", type=int, default=103500)
    parser.add_argument("--graphs", type=int, default=2)
    parser.add_argument("--rss-stop-gib", type=float, default=8.0)
    parser.add_argument("--output-name", default="v2_line_next_confirm_103500")
    args = parser.parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(","))
    families = tuple(args.families.split(","))
    layers = tuple(int(value) for value in args.layers.split(","))
    backends = tuple(args.backends.split(","))
    if (not sizes or any(n <= 24 or n > 64 for n in sizes)
            or not families or any(f not in FAMILIES for f in families)
            or not layers or any(p not in (1, 2) for p in layers)
            or not backends or any(name not in BACKENDS for name in backends)
            or args.graphs < 1 or args.rss_stop_gib <= 0
            or not args.output_name.replace("_", "").isalnum()):
        parser.error("Invalid case or resource limit")
    output = ROOT / "results" / f"{args.output_name}.csv"
    assert_fresh(args.graph_start, args.graphs, output)
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8"))) if output.exists() else []
    seen = {(r["snapshot"], int(r["logical_qubits"]), r["family"],
             int(r["graph_id"]), int(r["layers"])) for r in rows}
    if len(seen) != len(rows):
        raise AssertionError("Duplicate confirmation case")
    total = len(backends) * len(sizes) * len(families) * len(layers) * args.graphs
    for backend_name in backends:
        backend = BACKENDS[backend_name]()
        for n in sizes:
            for family in families:
                for graph_id in range(args.graph_start, args.graph_start + args.graphs):
                    seed = graph_id + 1_000_000 * n + (100_000 if family == "dense75" else 0)
                    edges = graph_edges(n, FAMILIES[family], seed)
                    for p in layers:
                        key = (backend.name, n, family, graph_id, p)
                        if key in seen:
                            continue
                        _, peak = process_memory_mib()
                        if peak is not None and peak > args.rss_stop_gib * 1024:
                            raise MemoryError(f"Stopped at {peak:.0f} MiB peak RAM")
                        print(f"[{len(rows)+1}/{total}] {backend.name} {n}q {family} "
                              f"graph={graph_id} p={p}", flush=True)
                        qc = qaoa_circuit(n, edges, p, seed + 211 + p)
                        order_started = perf_counter()
                        positions = optimise_line_layout_rust(qc, seed=11, max_seconds=0.15)
                        shared_order_seconds = perf_counter() - order_started
                        started = perf_counter()
                        frozen = compile_hybrid_dense_v2(
                            qc.copy(), backend, seed_transpiler=11,
                            line_positions=positions)
                        frozen_seconds = perf_counter() - started + shared_order_seconds
                        f_count, f_depth = native_metrics(frozen, backend)
                        started = perf_counter()
                        next_result, selected = compile_hybrid_line_next(
                            qc.copy(), backend, seed_transpiler=11,
                            return_candidate=True, seed11_positions=positions)
                        next_seconds = perf_counter() - started + shared_order_seconds
                        route_replay(qc, selected.routed, selected.layout, backend.coupling_map)
                        x_count, x_depth = native_metrics(next_result, backend)
                        output_positions = base._final_positions(next_result.circuit, backend.num_qubits)
                        mapping_ok = tuple(output_positions[q] for q in selected.routed.final_positions)
                        if mapping_ok != tuple(next_result.final_positions):
                            raise AssertionError("Native compiler changed the selected final mapping")
                        _, peak = process_memory_mib()
                        row = {"snapshot": backend.name, "logical_qubits": n,
                               "family": family, "graph_id": graph_id, "layers": p,
                               "frozen_2q": f_count, "next_2q": x_count,
                               "frozen_depth": f_depth, "next_depth": x_depth,
                               "frozen_seconds": frozen_seconds,
                               "next_seconds": next_seconds,
                               "next_route_mode": next_result.route_mode,
                               "route_check": "passed", "mapping_check": "passed",
                               "peak_rss_mib": peak}
                        rows.append(row)
                        seen.add(key)
                        with output.open("w", newline="", encoding="utf-8") as handle:
                            writer = csv.DictWriter(handle, fieldnames=FIELDS)
                            writer.writeheader()
                            writer.writerows(rows)
                        output.with_suffix(".md").write_text(report(rows, args), encoding="utf-8")
                        print(f"  frozen {f_count}/{f_depth}; next {x_count}/{x_depth} "
                              f"({next_result.route_mode})", flush=True)
    print(report(rows, args), flush=True)


if __name__ == "__main__":
    main()
