"""Local Stock Qiskit versus v2.0.0 on full 127-qubit IBM fake targets.

The input has 16 logical qubits, but Stock and Hybrid can use all 127 physical
sites. This is a compiler benchmark, not an IBM QPU run or a noisy simulation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKyiv, FakeSherbrooke

import benchmark as bench
import hybrid_level3 as v1
from hybrid_level3_v2 import LayoutSearchFailed, _error_cost, compare_variants


ROOT = Path(__file__).resolve().parent
BACKENDS = (FakeBrisbane, FakeKyiv, FakeSherbrooke)
FAMILIES = {"medium50", "dense75", "er50", "er75"}
METHODS = ("stock", "identity_rust", "stock_seed_rust", "multistart",
           "tournament", "guarded", "error_aware")
FIELDS = ("snapshot", "family", "graph_id", "layers", "method", "route_mode",
          "native_2q", "twoq_depth", "depth", "estimated_2q_error", "compile_seconds",
          "layouts_routed", "finalists_compiled", "route_state_error", "mps_max_probability_error")


def verify_routed(reference, routed, layout) -> float:
    """Remove SWAP bookkeeping and check the exact 16-logical-qubit state."""
    at = [None] * routed.circuit.num_qubits
    for logical, physical in enumerate(layout):
        at[physical] = logical
    replay = QuantumCircuit(len(layout), global_phase=routed.circuit.global_phase)
    for instruction in routed.circuit.data:
        physical = [routed.circuit.find_bit(q).index for q in instruction.qubits]
        if instruction.operation.name == "swap":
            a, b = physical
            at[a], at[b] = at[b], at[a]
            continue
        logical = [at[q] for q in physical]
        if any(q is None for q in logical):
            raise AssertionError("Router applied a gate to an empty physical site")
        replay.append(instruction.operation, logical)
    positions = tuple(at.index(logical) for logical in range(len(layout)))
    if positions != tuple(routed.final_positions):
        raise AssertionError("Router final-position map disagrees with its SWAPs")
    error = bench.amplitude_error(reference, Statevector.from_instruction(replay).data)
    if error > bench.STATE_TOL:
        raise AssertionError(f"Rust route changes the logical state: {error}")
    return error


def mps_probability_error(compiled, reference_probabilities, simulator) -> float:
    circuit = compiled.circuit.copy()
    circuit.save_probabilities(qubits=compiled.final_positions, label="logical_probabilities")
    result = simulator.run(circuit, shots=1).result()
    if not result.success:
        raise RuntimeError(f"MPS simulation failed: {result.status}")
    probabilities = result.data()["logical_probabilities"]
    error = float(np.max(np.abs(probabilities - reference_probabilities)))
    if error > 1e-8:
        raise AssertionError(f"Compiled probability check failed: {error}")
    return error


def selected_specs(graph_start: int, cases_per_family: int):
    return [spec for spec in bench.graph_specs("standard", graph_start)
            if spec.family in FAMILIES and spec.graph_id < graph_start + cases_per_family]


def run(output: Path, layouts: int, graph_start: int, cases_per_family: int,
        layers: tuple[int, ...], backend_names: tuple[str, ...], mps: bool,
        beam_seconds: float, stock_layout: bool):
    rows = []
    specs = selected_specs(graph_start, cases_per_family)
    simulator = AerSimulator(method="matrix_product_state") if mps else None
    for backend_class in BACKENDS:
        if backend_class.__name__ not in backend_names:
            continue
        backend = backend_class()
        for spec in specs:
            for depth in layers:
                print(f"{backend.name} {spec.family} {spec.graph_id} p={depth}", flush=True)
                logical, angles = bench.qaoa_circuit(spec.edges, depth, spec.graph_seed + 211 + depth)
                reference = bench.reference_statevector(spec.edges, angles)
                start = perf_counter()
                stock = v1.compile_stock_level3(logical.copy(), backend, seed_transpiler=11)
                stock_seconds = perf_counter() - start
                preferred = (tuple(stock.circuit.layout.initial_index_layout()[:logical.num_qubits])
                             if stock_layout else None)
                start = perf_counter()
                try:
                    variants, candidates = compare_variants(
                        logical.copy(), backend, seed_transpiler=11,
                        layouts=layouts, finalists=4, beam_seconds=beam_seconds,
                        preferred_layout=preferred,
                    )
                    v2_seconds = perf_counter() - start
                    route_error = max(verify_routed(reference, candidate.routed, candidate.layout)
                                      for candidate in candidates)
                except LayoutSearchFailed:
                    variants, candidates = {"multistart": stock, "tournament": stock,
                                            "error_aware": None}, []
                    v2_seconds = perf_counter() - start + stock_seconds
                    route_error = None
                identity = next((candidate.compiled for candidate in candidates
                                 if candidate.layout == tuple(range(logical.num_qubits))), None)
                stock_seed = next((candidate.compiled for candidate in candidates
                                   if candidate.layout == preferred), None)
                finalist_count = sum(candidate.compiled is not None for candidate in candidates)
                raw = variants["tournament"]
                raw_key = (sum(inst.operation.num_qubits == 2 for inst in raw.circuit.data),
                           bench.twoq_depth(raw.circuit))
                stock_key = (sum(inst.operation.num_qubits == 2 for inst in stock.circuit.data),
                             bench.twoq_depth(stock.circuit))
                guarded = stock if stock_key < raw_key else raw
                guarded_seconds = v2_seconds + (stock_seconds if candidates else 0.0)
                arms = (("stock", stock, stock_seconds), ("identity_rust", identity, None),
                        ("stock_seed_rust", stock_seed, None),
                        ("multistart", variants["multistart"], None),
                        ("tournament", variants["tournament"], v2_seconds),
                        ("guarded", guarded, guarded_seconds),
                        ("error_aware", variants["error_aware"], None))
                check_mps = (simulator is not None and spec.graph_id == graph_start and depth == 1)
                ideal_probabilities = np.abs(reference) ** 2 if check_mps else None
                for method, compiled, seconds in arms:
                    if compiled is None:
                        continue
                    mps_error = (mps_probability_error(compiled, ideal_probabilities, simulator)
                                 if check_mps and method in {"stock", "tournament"} else None)
                    rows.append({
                        "snapshot": backend.name, "family": spec.family,
                        "graph_id": spec.graph_id, "layers": depth, "method": method,
                        "route_mode": ("v2_failed_stock_fallback" if not candidates and method in
                                       {"multistart", "tournament", "guarded"} else
                                       "v2_stock_quality_guard" if method == "guarded" and guarded is stock
                                       and guarded is not raw else compiled.route_mode),
                        "native_2q": sum(inst.operation.num_qubits == 2 for inst in compiled.circuit.data),
                        "twoq_depth": bench.twoq_depth(compiled.circuit),
                        "depth": compiled.circuit.depth(),
                        "estimated_2q_error": _error_cost(compiled.circuit, backend.target),
                        "compile_seconds": seconds if seconds is not None else "",
                        "layouts_routed": len(candidates) if method not in {"stock"} else "",
                        "finalists_compiled": finalist_count if method in {"tournament", "guarded", "error_aware"} else "",
                        "route_state_error": route_error if method not in {"stock"} else "",
                        "mps_max_probability_error": mps_error if mps_error is not None else "",
                    })
                with output.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
    report(output, rows, layouts, graph_start, cases_per_family, layers, beam_seconds, stock_layout)


def report(output: Path, rows, layouts: int, graph_start: int,
           cases_per_family: int, layers: tuple[int, ...], beam_seconds: float,
           stock_layout: bool):
    key = lambda row: (row["snapshot"], row["family"], row["graph_id"], row["layers"])
    stock_times = {key(row): float(row["compile_seconds"])
                   for row in rows if row["method"] == "stock"}
    lines = ["# Full-target IBM snapshot routing research", "",
             "Each input has 16 logical qubits. Stock Qiskit Level 3 and v2.0.0 both receive "
             "the complete 127-qubit IBM fake target, with the same workload and seed 11.",
             "These are local compilation results, not QPU measurements or noisy simulations.",
             f"Fixed workload: {cases_per_family} graph(s) per family, families medium50/dense75/er50/er75, "
             f"layers {','.join(map(str, layers))}, graph ids starting at {graph_start}.",
             f"v2.0.0 tried {layouts} starting layout(s) and fully compiled at most four finalists. "
             "Additional layouts are ranked by graph distance before Rust routing.",
             ("One v2.0.0 start uses Stock's saved initial layout from the same Level-3 run. "
              "This is a Qiskit-assisted Hybrid, so Stock compile time is included in guarded time."
              if stock_layout else "v2.0.0 starts do not use Stock's initial layout."),
             f"Each Rust layout had a {beam_seconds:g} s search limit. The guarded arm includes Stock as a final quality check; "
             "report raw tournament numbers when judging the Rust router itself.",
             "Rust routes were exactly checked after reducing SWAP motion to the 16 logical qubits. "
             "Where requested, selected fully compiled circuits were also checked with 127-qubit MPS output probabilities. "
             "Probability checks do not establish full phase equivalence of the compiled circuit.",
             "Estimated 2Q error is sum(-log(1-p)) for native two-qubit gates when all saved edge errors exist. "
             "It is not measured QPU fidelity.",
             "Rows with different case counts have different coverage: their total gates and mean depth "
             "must not be compared directly. The paired comparisons below use identical workloads and targets.",
             "Median tournament time includes the Stock run needed to obtain its starting layout "
             "when Stock seeding is enabled. Guarded time also includes this run.", ""]
    for name in ("all", "fake_brisbane", "fake_kyiv", "fake_sherbrooke"):
        subset = rows if name == "all" else [row for row in rows if row["snapshot"] == name]
        if not subset:
            continue
        lines.extend([f"## {name}", "", "| Method | Cases | Total native 2Q | Mean 2Q depth | Median end-to-end s | Mean estimated 2Q error |",
                      "|---|---:|---:|---:|---:|---:|"])
        for method in METHODS:
            group = [row for row in subset if row["method"] == method]
            if not group:
                continue
            errors = [float(row["estimated_2q_error"]) for row in group
                      if row["estimated_2q_error"] not in (None, "")]
            error_text = f"{sum(errors)/len(errors):.3f}" if len(errors) == len(group) else "n/a"
            times = [float(row["compile_seconds"]) for row in group if row["compile_seconds"] not in (None, "")]
            if method == "tournament" and stock_layout:
                times = [float(row["compile_seconds"]) +
                         (0.0 if row["route_mode"] == "v2_failed_stock_fallback" else stock_times[key(row)])
                         for row in group if row["compile_seconds"] not in (None, "")]
            time_text = f"{median(times):.3f}" if len(times) == len(group) else "n/a"
            lines.append(f"| {method} | {len(group)} | {sum(int(row['native_2q']) for row in group)} | "
                         f"{sum(int(row['twoq_depth']) for row in group)/len(group):.1f} | "
                         f"{time_text} | {error_text} |")
        lines.append("")
    by_method = {
        method: {(row["snapshot"], row["family"], row["graph_id"], row["layers"]): row
                 for row in rows if row["method"] == method}
        for method in METHODS
    }
    lines.extend(["## Paired comparisons", "",
                  "Each line uses only the cases present in both methods. Wins/ties/losses count "
                  "native 2Q gates for the second method against the first.", "",
                  "| Baseline | Variant | Paired cases | Baseline 2Q | Variant 2Q | Change | Wins/ties/losses | Paired estimated-error wins/ties/losses |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for baseline, variant in (("stock", "tournament"), ("stock", "guarded"),
                              ("stock", "stock_seed_rust"), ("stock", "error_aware"),
                              ("tournament", "error_aware")):
        keys = sorted(by_method[baseline].keys() & by_method[variant].keys())
        if not keys:
            continue
        base = [by_method[baseline][key] for key in keys]
        trial = [by_method[variant][key] for key in keys]
        base_gates = sum(int(row["native_2q"]) for row in base)
        trial_gates = sum(int(row["native_2q"]) for row in trial)
        gate_differences = [int(a["native_2q"]) - int(b["native_2q"])
                            for a, b in zip(base, trial)]
        gate_record = (sum(delta > 0 for delta in gate_differences),
                       sum(delta == 0 for delta in gate_differences),
                       sum(delta < 0 for delta in gate_differences))
        error_pairs = [(float(a["estimated_2q_error"]), float(b["estimated_2q_error"]))
                       for a, b in zip(base, trial)
                       if a["estimated_2q_error"] not in (None, "")
                       and b["estimated_2q_error"] not in (None, "")]
        error_record = (sum(a > b + 1e-12 for a, b in error_pairs),
                        sum(abs(a - b) <= 1e-12 for a, b in error_pairs),
                        sum(a < b - 1e-12 for a, b in error_pairs))
        lines.append(f"| {baseline} | {variant} | {len(keys)} | {base_gates} | "
                     f"{trial_gates} | {(trial_gates / base_gates - 1) * 100:+.1f}% | "
                     f"{'/'.join(map(str, gate_record))} | "
                     f"{'/'.join(map(str, error_record))} ({len(error_pairs)} pairs) |")
    lines.append("")
    lines.extend(["## Raw tournament by graph family", "",
                  "Each graph/layer workload is compiled on three distinct 127-qubit snapshot targets. "
                  "These three observations share the logical workload and are not fully independent.", "",
                  "| Family | Paired cases | Stock 2Q | Raw Hybrid 2Q | Change | Wins/ties/losses |",
                  "|---|---:|---:|---:|---:|---:|"])
    for family in sorted({row["family"] for row in rows}):
        keys = [item for item in by_method["stock"].keys() & by_method["tournament"].keys()
                if item[1] == family]
        base = [int(by_method["stock"][item]["native_2q"]) for item in keys]
        trial = [int(by_method["tournament"][item]["native_2q"]) for item in keys]
        wins = sum(a > b for a, b in zip(base, trial))
        ties = sum(a == b for a, b in zip(base, trial))
        losses = sum(a < b for a, b in zip(base, trial))
        lines.append(f"| {family} | {len(keys)} | {sum(base)} | {sum(trial)} | "
                     f"{(sum(trial) / sum(base) - 1) * 100:+.1f}% | "
                     f"{wins}/{ties}/{losses} |")
    lines.append("")
    failures = sum(row["route_mode"] == "v2_failed_stock_fallback" for row in rows
                   if row["method"] == "tournament")
    mps_errors = [float(row["mps_max_probability_error"]) for row in rows
                  if row["mps_max_probability_error"] not in (None, "")]
    route_errors = [float(row["route_state_error"]) for row in rows
                    if row["method"] == "tournament" and row["route_state_error"] not in (None, "")]
    lines.extend([f"v2.0.0 fell back to Stock on {failures} cases.",
                  f"Exact Rust-route checks: {len(route_errors)} cases; maximum state error "
                  f"{max(route_errors) if route_errors else 'n/a'}.",
                  f"MPS compiled-output checks: {len(mps_errors)} circuits; maximum probability error "
                  f"{max(mps_errors) if mps_errors else 'n/a'}.",
                  f"Per-case results: {output.name}."])
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=int, default=8)
    parser.add_argument("--graph-start", type=int, default=84000)
    parser.add_argument("--cases-per-family", type=int, default=1)
    parser.add_argument("--layers", default="1")
    parser.add_argument("--backends", default="FakeBrisbane,FakeKyiv,FakeSherbrooke")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--beam-seconds", type=float, default=5.0)
    parser.add_argument("--stock-layout", action="store_true")
    parser.add_argument("--output-name", default="v2_ibm_full_8")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (not args.output_name.startswith("v2_ibm_full_")
            or Path(args.output_name).name != args.output_name):
        parser.error("--output-name must be a plain v2_ibm_full_* prefix")
    if args.cases_per_family < 1 or args.layouts < 1 or args.beam_seconds <= 0:
        parser.error("Cases per family, layouts, and beam seconds must be positive")
    try:
        layers = tuple(int(value) for value in args.layers.split(","))
    except ValueError:
        parser.error("--layers must be comma-separated integers")
    if any(layer not in (1, 2) for layer in layers):
        parser.error("--layers supports 1 and 2")
    names = tuple(value.strip() for value in args.backends.split(","))
    if not set(names).issubset({backend.__name__ for backend in BACKENDS}):
        parser.error("Unknown backend in --backends")
    output = ROOT / "results" / f"{args.output_name}.csv"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Results already exist: {output}")
    run(output, args.layouts, args.graph_start, args.cases_per_family, layers,
        names, args.mps, args.beam_seconds, args.stock_layout)
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
