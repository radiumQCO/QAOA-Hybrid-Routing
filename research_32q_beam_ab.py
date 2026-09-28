"""A/B test beam vs no beam on the existing 32q paired circuits."""
from __future__ import annotations

import csv
import argparse
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKyiv, FakeSherbrooke

import hybrid_level3 as base
from hybrid_level3_v2 import compile_hybrid_dense_v2
from research_scaling import graph_edges, native_metrics, process_memory_mib, qaoa_circuit, route_replay
from rust_router_core import optimise_line_layout_rust


ROOT = Path(__file__).resolve().parent
INPUTS = (ROOT / "results/v2_32q_fresh_20260928.csv",
          ROOT / "results/v2_32q_confirm_20260928.csv")
OUTPUT = ROOT / "results/v2_32q_beam_ab_20260928.csv"
REPORT = OUTPUT.with_suffix(".md")
BACKENDS = {"fake_brisbane": FakeBrisbane, "fake_kyiv": FakeKyiv,
            "fake_sherbrooke": FakeSherbrooke}
FIELDS = ("snapshot", "family", "graph_id", "layers", "edges",
          "with_2q", "without_2q", "with_depth", "without_depth",
          "with_seconds", "without_seconds", "order_seconds", "with_mode",
          "without_mode", "with_route_checks", "without_route_checks",
          "peak_rss_mib")


def key(row):
    return (row["snapshot"], row["family"], int(row["graph_id"]), int(row["layers"]))


def save(rows):
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def report(rows):
    lines = ["# 32q beam ablation", "",
             "A paired A/B on the *existing* 32q benchmark circuits (graph IDs "
             "99600–99604 and 99700–99704). These are deliberately reused for "
             "a controlled ablation, not presented as fresh holdout graphs. "
             "Both variants use the exact same Rust-optimized line ordering per "
             "case, the same Qiskit seed and target, and symbolic route checks. "
             "The ordering search time is added to both compile times. "
             "No full statevector or QPU execution was used.", "",
             "| Family | Cases | Equal 2Q/depth | With beam routes | Without beam routes | "
             "Mean with beam | Mean without beam | Median time saved |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, subset in (("All", rows), ("medium50", [r for r in rows if r["family"] == "medium50"]),
                          ("dense75", [r for r in rows if r["family"] == "dense75"])):
        if not subset:
            continue
        equal = sum(int(r["with_2q"]) == int(r["without_2q"])
                    and int(r["with_depth"]) == int(r["without_depth"]) for r in subset)
        with_routes = sum(r["with_mode"].startswith("v2_rust") for r in subset)
        without_routes = sum(r["without_mode"].startswith("v2_rust") for r in subset)
        with_mean = mean(float(r["with_seconds"]) for r in subset)
        without_mean = mean(float(r["without_seconds"]) for r in subset)
        saved = median(float(r["with_seconds"]) - float(r["without_seconds"]) for r in subset)
        lines.append(f"| {label} | {len(subset)} | {equal} | {with_routes} | "
                     f"{without_routes} | {with_mean:.3f} s | {without_mean:.3f} s | "
                     f"{saved:.3f} s |")
    if rows:
        lines += ["", "Both variants passed symbolic SWAP/gate-stream replay, "
                  "selected final-mapping checks, coupling validation, and native "
                  "2Q gate checks in every recorded case.", "",
                  f"Beam completed in {sum(int(r['with_route_checks']) > int(r['without_route_checks']) for r in rows)} "
                  f"of {len(rows)} cases. Peak process RAM: "
                  f"{max(float(r['peak_rss_mib']) for r in rows):.0f} MiB."]
    return "\n".join(lines) + "\n"


def compile_variant(qc, backend, positions, beam_seconds):
    certificates = {}

    def verify(name, reference, routed, layout, cmap):
        route_replay(reference, routed, layout, cmap)
        certificates[name] = tuple(routed.final_positions)

    started = perf_counter()
    result = compile_hybrid_dense_v2(
        qc.copy(), backend, seed_transpiler=11, beam_seconds=beam_seconds,
        line_positions=positions, route_verifier=verify)
    elapsed = perf_counter() - started
    twoq, depth = native_metrics(result, backend)
    if not result.route_mode.startswith("v2_rust"):
        raise AssertionError("A/B variant fell back to Stock")
    name = next((candidate for candidate in certificates
                 if result.route_mode.endswith("_" + candidate)), None)
    if name is None:
        raise AssertionError("Selected candidate has no symbolic certificate")
    physical_output = base._final_positions(result.circuit, backend.num_qubits)
    expected = tuple(physical_output[q] for q in certificates[name])
    if expected != tuple(result.final_positions):
        raise AssertionError("Compiled mapping does not match symbolic route")
    return twoq, depth, elapsed, result.route_mode, len(certificates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.limit <= 120:
        parser.error("--limit must be between 1 and 120")
    source = [row for path in INPUTS for row in csv.DictReader(path.open(encoding="utf-8"))]
    if len(source) != 120 or len({key(row) for row in source}) != 120:
        raise AssertionError("Expected 120 unique existing 32q benchmark cases")
    rows = list(csv.DictReader(OUTPUT.open(encoding="utf-8"))) if OUTPUT.exists() else []
    seen = {key(row) for row in rows}
    if len(seen) != len(rows):
        raise AssertionError("Duplicate A/B output rows")
    backend_cache = {name: cls() for name, cls in BACKENDS.items()}
    for index, spec in enumerate(source[:args.limit]):
        identity = key(spec)
        if identity in seen:
            continue
        backend_name, family, graph_id, layers = identity
        backend = backend_cache[backend_name]
        seed = graph_id + 32_000_000 + (100_000 if family == "dense75" else 0)
        edges = graph_edges(32, 0.75 if family == "dense75" else 0.50, seed)
        qc = qaoa_circuit(32, edges, layers, seed + 211 + layers)
        started = perf_counter()
        positions = optimise_line_layout_rust(qc, seed=11, max_seconds=0.15)
        order_seconds = perf_counter() - started
        print(f"[{len(rows) + 1}/120] {backend_name} {family} "
              f"graph={graph_id} p={layers}", flush=True)
        variant_order = ("with", "without") if index % 2 == 0 else ("without", "with")
        variants = {}
        for name in variant_order:
            variants[name] = compile_variant(qc, backend, positions, 1.0 if name == "with" else 0.0)
        _, peak = process_memory_mib()
        row = {"snapshot": backend_name, "family": family, "graph_id": graph_id,
               "layers": layers, "edges": len(edges), "order_seconds": order_seconds,
               "peak_rss_mib": peak}
        for name, (twoq, depth, seconds, mode, checks) in variants.items():
            row[f"{name}_2q"] = twoq
            row[f"{name}_depth"] = depth
            row[f"{name}_seconds"] = seconds + order_seconds
            row[f"{name}_mode"] = mode
            row[f"{name}_route_checks"] = checks
        rows.append(row)
        seen.add(identity)
        save(rows)
        REPORT.write_text(report(rows), encoding="utf-8")
    final = report(rows)
    REPORT.write_text(final, encoding="utf-8")
    print(final, flush=True)


if __name__ == "__main__":
    main()
