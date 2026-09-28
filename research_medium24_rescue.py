"""Paired local check of the 24q medium route against the frozen alternatives."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

from qiskit.circuit.library import PauliEvolutionGate
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKyiv, FakeSherbrooke

import hybrid_level3 as base
from hybrid_level3_v2 import compile_hybrid_dense_v2, compile_hybrid_level3_v2
from layout_seed import physical_line
from research_commuting_baseline import compile_commuting
from research_final_holdout import assert_fresh
from research_scaling import graph_edges, native_metrics, qaoa_circuit, route_replay
from rust_router_core import optimise_line_layout_rust, route_line_layers_rust


ROOT = Path(__file__).resolve().parent
BACKENDS = (FakeBrisbane, FakeKyiv, FakeSherbrooke)
FIELDS = (
    "snapshot", "graph_id", "layers", "edges", "stock_2q", "old_2q",
    "new_2q", "commuting_2q", "stock_depth", "old_depth", "new_depth",
    "commuting_depth", "stock_seconds", "old_seconds", "new_seconds",
    "commuting_seconds", "old_mode", "new_mode", "line_replay",
)


def save(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summary(rows: list[dict], ids: range) -> str:
    lines = [
        "# 24q medium route experiment", "",
        f"Fresh graph IDs {ids.start}–{ids.stop - 1}; 24 logical qubits, density 0.50, "
        f"{len(ids) * 2} distinct circuits (two QAOA layer counts per graph) on "
        "three full IBM fake target snapshots. This is "
        "local compilation, not QPU execution. The previous 98000–98004 final "
        "holdout was not reused.", "",
        "The new Hybrid tries two complete Rust line routes and the existing beam "
        "route. It chooses the circuit with the fewest native 2Q gates among "
        "candidates within 10% of the smallest 2Q depth. Stock, old Hybrid, "
        "and Qiskit commuting + SAT are compiled on the same circuits. A Stock "
        "fallback counts as a Hybrid routing failure.", "",
        "| Group | Cases | Old routes | New routes | Stock 2Q | Old 2Q | "
        "New 2Q | Commuting 2Q | Old depth | New depth | Commuting depth | "
        "New vs commuting |", "| --- | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, subset in (("All", rows), ("p=1", [r for r in rows if int(r["layers"]) == 1]),
                          ("p=2", [r for r in rows if int(r["layers"]) == 2])):
        if not subset:
            continue
        total = lambda name: sum(int(r[name]) for r in subset)
        old_routes = sum(str(r["old_mode"]).startswith("v2_rust") for r in subset)
        new_routes = sum(str(r["new_mode"]).startswith("v2_rust") for r in subset)
        change = 100 * (total("new_2q") / total("commuting_2q") - 1)
        lines.append(f"| {label} | {len(subset)} | {old_routes} | {new_routes} | "
                     f"{total('stock_2q')} | {total('old_2q')} | {total('new_2q')} | "
                     f"{total('commuting_2q')} | {total('old_depth')} | "
                     f"{total('new_depth')} | {total('commuting_depth')} | {change:+.1f}% |")
    if rows:
        wins = sum(int(r["new_2q"]) < int(r["commuting_2q"]) for r in rows)
        ties = sum(int(r["new_2q"]) == int(r["commuting_2q"]) for r in rows)
        losses = len(rows) - wins - ties
        lines += ["", f"New Hybrid vs commuting per-case 2Q: {wins} wins, {ties} ties, "
                  f"{losses} losses. Symbolic line replay passed "
                  f"{sum(r['line_replay'] == 'passed' for r in rows)}/{len(rows)} "
                  "cases. Native target instruction checks passed for every recorded circuit.", ""]
        for name in ("stock", "old", "new", "commuting"):
            mean = sum(float(r[f"{name}_seconds"]) for r in rows) / len(rows)
            lines.append(f"Mean {name} compilation: {mean:.3f} s.")
        lines += ["", "No 24-qubit statevector was created. Symbolic replay checks "
                  "the line route before native compilation; it is not a full "
                  "quantum simulation of every compiled circuit."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-start", type=int, default=99000)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output-name", default="v2_medium24_rescue_20260928")
    args = parser.parse_args()
    if args.count < 1 or not args.output_name.replace("_", "").isalnum():
        parser.error("Use a positive count and a simple output name")
    output = ROOT / "results" / (args.output_name + ".csv")
    report = ROOT / "results" / (args.output_name + ".md")
    ids = range(args.graph_start, args.graph_start + args.count)
    assert_fresh(args.graph_start, args.count, output)
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8"))) if output.exists() else []
    seen = {(r["snapshot"], int(r["graph_id"]), int(r["layers"])) for r in rows}
    if len(seen) != len(rows):
        raise AssertionError("Duplicate output rows")
    total = args.count * 2 * len(BACKENDS)
    for backend_class in BACKENDS:
        backend = backend_class()
        path = physical_line(backend.coupling_map, 24)
        for graph_id in ids:
            seed = graph_id + 1_000_000 * 24
            edges = graph_edges(24, 0.50, seed)
            for layers in (1, 2):
                key = (backend.name, graph_id, layers)
                if key in seen:
                    continue
                print(f"[{len(rows) + 1}/{total}] {backend.name} 24q medium50 "
                      f"graph={graph_id} p={layers}", flush=True)
                qc = qaoa_circuit(24, edges, layers, seed + 211 + layers)
                compiled = {}
                seconds = {}
                for name, function in (
                    ("stock", lambda: base.compile_stock_level3(qc.copy(), backend, seed_transpiler=11)),
                    ("old", lambda: compile_hybrid_level3_v2(
                        qc.copy(), backend, seed_transpiler=11, layouts=1,
                        beam_seconds=1.0, quality_guard=False)),
                    ("new", lambda: compile_hybrid_dense_v2(qc.copy(), backend, seed_transpiler=11)),
                ):
                    started = perf_counter()
                    compiled[name] = function()
                    seconds[name] = perf_counter() - started
                first = next(inst for inst in qc.data
                             if isinstance(inst.operation, PauliEvolutionGate))
                pairs = tuple(sorted(base._layer_terms(qc, first)))
                started = perf_counter()
                compiled["commuting"], _, _, _, _, _, _, _ = compile_commuting(
                    qc, backend, pairs, path, 11, 2.0)
                seconds["commuting"] = perf_counter() - started
                metrics = {name: native_metrics(result, backend)
                           for name, result in compiled.items()}
                order = optimise_line_layout_rust(qc, seed=11, max_seconds=0.15)
                layout = tuple(path[position] for position in order)
                routed = route_line_layers_rust(qc.copy(), backend.coupling_map, path, layout)
                route_replay(qc, routed, layout, backend.coupling_map)
                row = {"snapshot": backend.name, "graph_id": graph_id,
                       "layers": layers, "edges": len(edges),
                       "old_mode": compiled["old"].route_mode,
                       "new_mode": compiled["new"].route_mode,
                       "line_replay": "passed"}
                for name in compiled:
                    row[f"{name}_2q"], row[f"{name}_depth"] = metrics[name]
                    row[f"{name}_seconds"] = seconds[name]
                rows.append(row)
                seen.add(key)
                save(output, rows)
                report.write_text(summary(rows, ids), encoding="utf-8")
    final_report = summary(rows, ids)
    report.write_text(final_report, encoding="utf-8")
    print(final_report, flush=True)


if __name__ == "__main__":
    main()
