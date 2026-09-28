"""Small memory-safe Hybrid-only smoke tests beyond 32 logical qubits."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from time import perf_counter

from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKyiv, FakeSherbrooke

import hybrid_level3 as base
from hybrid_level3_v2 import compile_hybrid_dense_v2
from research_final_holdout import assert_fresh
from research_scaling import graph_edges, native_metrics, process_memory_mib, qaoa_circuit, route_replay


ROOT = Path(__file__).resolve().parent
BACKENDS = {"fake_brisbane": FakeBrisbane, "fake_kyiv": FakeKyiv,
            "fake_sherbrooke": FakeSherbrooke}
DENSITIES = {"medium50": 0.50, "dense75": 0.75}
FIELDS = ("snapshot", "logical_qubits", "family", "graph_id", "layers", "edges",
          "native_2q", "twoq_depth", "compile_seconds", "route_mode",
          "route_checks", "mapping_check", "peak_rss_mib")


def save(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def report(rows, args):
    lines = [f"# {args.size}q Hybrid smoke test", "",
             f"Fresh graph IDs {args.graph_start}–{args.graph_start + args.graphs - 1}; "
             "FakeBrisbane, medium50 and dense75, one and two QAOA layers. "
             "Hybrid uses no beam above 24q. This is a small local compilation "
             "and correctness check, not a comparison with Stock or Qiskit "
             "commuting + SAT and not a real-QPU run. No statevector or MPS "
             "simulation is used.", "",
             "| Family | Cases | Routes | Native 2Q | 2Q depth | Mean compile time | Peak RAM |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for family in DENSITIES:
        subset = [r for r in rows if r["family"] == family]
        if not subset:
            continue
        routes = sum(r["route_mode"].startswith("v2_rust") for r in subset)
        lines.append(f"| {family} | {len(subset)} | {routes} | "
                     f"{sum(int(r['native_2q']) for r in subset)} | "
                     f"{sum(int(r['twoq_depth']) for r in subset)} | "
                     f"{mean(float(r['compile_seconds']) for r in subset):.2f} s | "
                     f"{max(float(r['peak_rss_mib']) for r in subset):.0f} MiB |")
    if rows:
        lines += ["", f"Symbolic route checks: {sum(int(r['route_checks']) for r in rows)}; "
                  f"selected compiled mapping checks: "
                  f"{sum(r['mapping_check'] == 'passed' for r in rows)}/{len(rows)}. "
                  "Symbolic replay validates SWAPs, ZZ gate order, final logical "
                  "placement, and coupling edges. Every compiled 2Q instruction "
                  "was checked against the target."]
    return "\n".join(lines) + "\n"


def compile_one(qc, backend):
    certificates = {}

    def verify(name, reference, routed, layout, cmap):
        route_replay(reference, routed, layout, cmap)
        certificates[name] = tuple(routed.final_positions)

    started = perf_counter()
    compiled = compile_hybrid_dense_v2(qc.copy(), backend, seed_transpiler=11,
                                       route_verifier=verify)
    elapsed = perf_counter() - started
    count, depth = native_metrics(compiled, backend)
    if not compiled.route_mode.startswith("v2_rust"):
        raise AssertionError(f"Hybrid did not complete a Rust route: {compiled.route_mode}")
    name = next((candidate for candidate in certificates
                 if compiled.route_mode.endswith("_" + candidate)), None)
    if name is None:
        raise AssertionError("Selected route has no symbolic certificate")
    physical_output = base._final_positions(compiled.circuit, backend.num_qubits)
    if tuple(physical_output[q] for q in certificates[name]) != tuple(compiled.final_positions):
        raise AssertionError("Compiled final mapping differs from Rust route")
    _, peak = process_memory_mib()
    return {"native_2q": count, "twoq_depth": depth, "compile_seconds": elapsed,
            "route_mode": compiled.route_mode, "route_checks": len(certificates),
            "mapping_check": "passed", "peak_rss_mib": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, required=True, choices=(40, 48, 64))
    parser.add_argument("--graph-start", type=int, required=True)
    parser.add_argument("--graphs", type=int, default=2)
    parser.add_argument("--rss-stop-gib", type=float, default=12.0)
    parser.add_argument("--output-name")
    args = parser.parse_args()
    if args.graphs < 1 or args.rss_stop_gib <= 0:
        parser.error("Expected positive graph count and RAM stop limit")
    if args.output_name is None:
        args.output_name = f"v2_{args.size}q_smoke_{args.graph_start}"
    if not args.output_name.replace("_", "").isalnum():
        parser.error("Use a simple output name")
    output = ROOT / "results" / (args.output_name + ".csv")
    report_path = output.with_suffix(".md")
    assert_fresh(args.graph_start, args.graphs, output)
    rows = list(csv.DictReader(output.open(encoding="utf-8"))) if output.exists() else []
    seen = {(r["family"], int(r["graph_id"]), int(r["layers"])) for r in rows}
    if len(seen) != len(rows):
        raise AssertionError("Duplicate smoke-test row")
    backend = FakeBrisbane()
    total = args.graphs * len(DENSITIES) * 2
    for family, density in DENSITIES.items():
        for graph_id in range(args.graph_start, args.graph_start + args.graphs):
            seed = graph_id + 1_000_000 * args.size + (100_000 if family == "dense75" else 0)
            edges = graph_edges(args.size, density, seed)
            for layers in (1, 2):
                identity = (family, graph_id, layers)
                if identity in seen:
                    continue
                _, peak = process_memory_mib()
                if peak is not None and peak > args.rss_stop_gib * 1024:
                    raise MemoryError(f"Stopping at {peak:.0f} MiB peak process RAM")
                print(f"[{len(rows) + 1}/{total}] {args.size}q {family} "
                      f"graph={graph_id} p={layers}", flush=True)
                qc = qaoa_circuit(args.size, edges, layers, seed + 211 + layers)
                row = {"snapshot": backend.name, "logical_qubits": args.size,
                       "family": family, "graph_id": graph_id, "layers": layers,
                       "edges": len(edges)}
                row.update(compile_one(qc, backend))
                rows.append(row)
                seen.add(identity)
                save(output, rows)
                report_path.write_text(report(rows, args), encoding="utf-8")
    final = report(rows, args)
    report_path.write_text(final, encoding="utf-8")
    print(final, flush=True)


if __name__ == "__main__":
    main()
