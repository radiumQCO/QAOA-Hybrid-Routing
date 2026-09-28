"""Memory-safe QAOA routing smoke test and paired 32–64q benchmark."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

from qiskit.circuit.library import PauliEvolutionGate
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKyiv, FakeSherbrooke

import hybrid_level3 as base
from hybrid_level3_v2 import compile_hybrid_dense_v2
from layout_seed import physical_line
from research_commuting_baseline import compile_commuting
from research_final_holdout import assert_fresh
from research_scaling import graph_edges, native_metrics, process_memory_mib, qaoa_circuit, route_replay


ROOT = Path(__file__).resolve().parent
BACKENDS = {"fake_brisbane": FakeBrisbane, "fake_kyiv": FakeKyiv,
            "fake_sherbrooke": FakeSherbrooke}
DENSITIES = {"medium50": 0.50, "dense75": 0.75}
FIELDS = ("snapshot", "logical_qubits", "family", "graph_id", "layers", "edges",
          "stock_2q", "hybrid_2q", "commuting_2q", "stock_depth", "hybrid_depth",
          "commuting_depth", "stock_seconds", "hybrid_seconds", "commuting_seconds",
          "sat_seconds", "sat_optimal", "hybrid_mode", "route_checks",
          "selected_mapping_check", "peak_rss_mib")


def save(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def report(rows: list[dict], args) -> str:
    lines = ["# 32q+ QAOA routing benchmark", "",
             f"Graph IDs {args.graph_start}–{args.graph_start + args.graphs - 1}; "
             "fresh MaxCut graph seeds, full IBM fake targets. This is local "
             "compilation, not execution on a QPU. No full statevector or MPS "
             "simulation is used.", "",
             "Symbolic replay checks each Rust route's SWAPs, gate order, final "
             "mapping, and coupling edges. The selected route's mapping is also "
             "checked after native compilation. Every 2Q gate is checked against "
             "the target. This does not prove full compiled-output quantum equivalence.", "",
             "| Size / graph | Cases | Hybrid routes | Stock 2Q | Hybrid 2Q | "
             "Commuting 2Q | Hybrid depth | Commuting depth | Hybrid vs commuting | "
             "W/T/L | Mean Hybrid / commuting time | Peak RAM |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
             "---: | ---: | ---: |"]
    groups = sorted({(int(r["logical_qubits"]), r["family"]) for r in rows})
    for n, family in groups:
        subset = [r for r in rows if int(r["logical_qubits"]) == n and r["family"] == family]
        total = lambda key: sum(int(r[key]) for r in subset)
        complete = sum(r["hybrid_mode"].startswith("v2_rust") for r in subset)
        w = sum(int(r["hybrid_2q"]) < int(r["commuting_2q"]) for r in subset)
        t = sum(int(r["hybrid_2q"]) == int(r["commuting_2q"]) for r in subset)
        l = len(subset) - w - t
        change = 100 * (total("hybrid_2q") / total("commuting_2q") - 1)
        ht = sum(float(r["hybrid_seconds"]) for r in subset) / len(subset)
        ct = sum(float(r["commuting_seconds"]) for r in subset) / len(subset)
        peak = max(float(r["peak_rss_mib"]) for r in subset)
        lines.append(f"| {n}q {family} | {len(subset)} | {complete} | "
                     f"{total('stock_2q')} | {total('hybrid_2q')} | "
                     f"{total('commuting_2q')} | {total('hybrid_depth')} | "
                     f"{total('commuting_depth')} | {change:+.1f}% | "
                     f"{w}/{t}/{l} | {ht:.2f} / {ct:.2f} s | {peak:.0f} MiB |")
    if rows:
        lines += ["", f"Recorded {len(rows)} paired cases. Symbolic route checks: "
                  f"{sum(int(r['route_checks']) for r in rows)} candidate routes; "
                  f"selected compiled mapping checks: "
                  f"{sum(r['selected_mapping_check'] == 'passed' for r in rows)}/{len(rows)}. "
                  "Target-native 2Q support checked for every recorded circuit."]
        sat_rows = [r for r in rows if r.get("sat_optimal") not in (None, "")]
        if sat_rows:
            proven = sum(str(r["sat_optimal"]).lower() == "true" for r in sat_rows)
            lines.append(f"SAT first-layer optimum proven in {proven}/{len(sat_rows)} "
                         "recorded cases; other cases retain the best feasible mapping "
                         "found under the fixed 2-second feasibility-check limit.")
    return "\n".join(lines) + "\n"


def compile_one(qc, backend, path, seed: int) -> dict:
    certificates = {}

    def verify(name, reference, routed, layout, cmap):
        route_replay(reference, routed, layout, cmap)
        certificates[name] = tuple(routed.final_positions)

    started = perf_counter()
    stock = base.compile_stock_level3(qc.copy(), backend, seed_transpiler=11)
    stock_seconds = perf_counter() - started
    started = perf_counter()
    hybrid = compile_hybrid_dense_v2(qc.copy(), backend, seed_transpiler=11,
                                     route_verifier=verify)
    hybrid_seconds = perf_counter() - started
    first = next(inst for inst in qc.data if isinstance(inst.operation, PauliEvolutionGate))
    pairs = tuple(sorted(base._layer_terms(qc, first)))
    started = perf_counter()
    commuting, _, _, _, sat_seconds, _, _, sat_optimal = compile_commuting(
        qc, backend, pairs, path, 11, 2.0)
    commuting_seconds = perf_counter() - started
    metrics = {name: native_metrics(result, backend) for name, result in
               (("stock", stock), ("hybrid", hybrid), ("commuting", commuting))}
    selected_check = "not_routed"
    if hybrid.route_mode.startswith("v2_rust"):
        name = next((candidate for candidate in certificates
                     if hybrid.route_mode.endswith("_" + candidate)), None)
        if name is None:
            raise AssertionError("Selected route has no symbolic replay certificate")
        physical_output = base._final_positions(hybrid.circuit, backend.num_qubits)
        expected = tuple(physical_output[q] for q in certificates[name])
        if expected != tuple(hybrid.final_positions):
            raise AssertionError("Selected compiled final mapping differs from routed mapping")
        selected_check = "passed"
    _, peak = process_memory_mib()
    row = {"stock_2q": metrics["stock"][0], "hybrid_2q": metrics["hybrid"][0],
           "commuting_2q": metrics["commuting"][0],
           "stock_depth": metrics["stock"][1], "hybrid_depth": metrics["hybrid"][1],
           "commuting_depth": metrics["commuting"][1],
           "stock_seconds": stock_seconds, "hybrid_seconds": hybrid_seconds,
           "commuting_seconds": commuting_seconds, "hybrid_mode": hybrid.route_mode,
           "sat_seconds": sat_seconds, "sat_optimal": sat_optimal,
           "route_checks": len(certificates), "selected_mapping_check": selected_check,
           "peak_rss_mib": peak}
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="One graph, both families and layer counts, on Brisbane")
    parser.add_argument("--sizes", default="32", help="Comma-separated sizes, later 40,48,64")
    parser.add_argument("--families", default="medium50,dense75")
    parser.add_argument("--backends", default="fake_brisbane,fake_kyiv,fake_sherbrooke")
    parser.add_argument("--layers", default="1,2")
    parser.add_argument("--graphs", type=int, default=5)
    parser.add_argument("--graph-start", type=int)
    parser.add_argument("--rss-stop-gib", type=float, default=12.0)
    parser.add_argument("--output-name")
    args = parser.parse_args()
    if args.smoke:
        args.families, args.backends, args.layers, args.graphs = (
            "medium50,dense75", "fake_brisbane", "1,2", 1)
    args.graph_start = args.graph_start if args.graph_start is not None else (99500 if args.smoke else 99600)
    if args.output_name is None:
        size_label = args.sizes.replace(",", "_")
        args.output_name = (f"v2_{size_label}q_smoke_20260928" if args.smoke
                            else f"v2_{size_label}q_fresh_20260928")
    sizes = [int(x) for x in args.sizes.split(",")]
    families = args.families.split(",")
    backends = args.backends.split(",")
    layers = [int(x) for x in args.layers.split(",")]
    if (not sizes or any(n not in (32, 40, 48, 64) for n in sizes)
            or any(f not in DENSITIES for f in families)
            or any(b not in BACKENDS for b in backends)
            or any(p not in (1, 2) for p in layers)
            or args.graphs < 1 or args.rss_stop_gib <= 0
            or not args.output_name.replace("_", "").isalnum()):
        parser.error("Invalid size, family, backend, layer, count, RAM limit, or output name")
    output = ROOT / "results" / (args.output_name + ".csv")
    report_path = ROOT / "results" / (args.output_name + ".md")
    assert_fresh(args.graph_start, args.graphs, output)
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8"))) if output.exists() else []
    seen = {(r["snapshot"], int(r["logical_qubits"]), r["family"],
             int(r["graph_id"]), int(r["layers"])) for r in rows}
    if len(seen) != len(rows):
        raise AssertionError("Duplicate output rows")
    total = len(sizes) * len(families) * len(backends) * len(layers) * args.graphs
    for backend_name in backends:
        backend = BACKENDS[backend_name]()
        paths = {n: physical_line(backend.coupling_map, n) for n in sizes}
        for n in sizes:
            for family in families:
                for graph_id in range(args.graph_start, args.graph_start + args.graphs):
                    seed = graph_id + 1_000_000 * n + (100_000 if family == "dense75" else 0)
                    edges = graph_edges(n, DENSITIES[family], seed)
                    for p in layers:
                        key = (backend.name, n, family, graph_id, p)
                        if key in seen:
                            continue
                        _, peak = process_memory_mib()
                        if peak is not None and peak > args.rss_stop_gib * 1024:
                            raise MemoryError(f"Stopping at {peak:.0f} MiB peak RAM")
                        print(f"[{len(rows) + 1}/{total}] {backend.name} {n}q "
                              f"{family} graph={graph_id} p={p}", flush=True)
                        qc = qaoa_circuit(n, edges, p, seed + 211 + p)
                        row = {"snapshot": backend.name, "logical_qubits": n,
                               "family": family, "graph_id": graph_id, "layers": p,
                               "edges": len(edges)}
                        row.update(compile_one(qc, backend, paths[n], seed))
                        rows.append(row)
                        seen.add(key)
                        save(output, rows)
                        report_path.write_text(report(rows, args), encoding="utf-8")
    final = report(rows, args)
    report_path.write_text(final, encoding="utf-8")
    print(final, flush=True)


if __name__ == "__main__":
    main()
