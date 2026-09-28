"""Compare dense line completion against the existing 348 paired compiler cases."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

from hybrid_level3_v2 import compile_hybrid_dense_v2
from research_no_reroute import cases
from research_scaling import native_metrics, process_memory_mib


ROOT = Path(__file__).resolve().parent
FIELDS = (
    "suite", "snapshot", "logical_qubits", "family", "graph_id", "layers", "edges",
    "stock_2q", "old_hybrid_2q", "dense_hybrid_2q", "commuting_2q",
    "stock_2q_depth", "old_hybrid_2q_depth", "dense_hybrid_2q_depth",
    "commuting_2q_depth", "old_hybrid_seconds", "dense_hybrid_seconds",
    "commuting_seconds", "old_hybrid_status", "dense_hybrid_mode", "peak_rss_mib",
)


def key(row) -> tuple[str, str, int, str, int, int]:
    return (row["suite"], row["snapshot"], int(row["logical_qubits"]),
            row["family"], int(row["graph_id"]), int(row["layers"]))


def references() -> dict[tuple, dict]:
    result = {}
    for name in ("v2_commuting_sat_holdout240.csv", "v2_commuting_sat_scaling108.csv"):
        with (ROOT / "results" / name).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                identity = key(row)
                if identity in result:
                    raise AssertionError(f"Duplicate baseline case: {identity}")
                result[identity] = row
    if len(result) != 348:
        raise AssertionError(f"Expected 348 baseline cases, found {len(result)}")
    return result


def save(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def report(path: Path, rows: list[dict]) -> None:
    snapshots = ", ".join(sorted({row["snapshot"] for row in rows}))
    logical_cases = len({(r["suite"], r["logical_qubits"], r["family"],
                          r["graph_id"], r["layers"]) for r in rows})
    lines = [
        "# Dense completion on IBM fake targets", "",
        f"Paired MaxCut circuits on full targets: {snapshots}. "
        "Stock, frozen Hybrid v2.0.0, and commuting + SAT counts come from the previous "
        "348-case run. The new Hybrid uses a complete Rust line route, a 0.15 s Rust "
        "search for a better line ordering, and the existing 1 s Rust beam candidate. "
        "It fully compiles the available candidates and selects the fewest native 2Q "
        "gates. Medium-density cases still use the existing Hybrid path. "
        "This is a local compiler test, not a QPU result.", "",
        f"The same logical workloads are repeated across backends: {len(rows)} compiler cases "
        f"cover {logical_cases} distinct logical circuits. Previous-run timings and this run's timings "
        "are indicative, not a simultaneous "
        "wall-clock race. A routed case means Rust produced a complete route without "
        "Stock fallback.", "",
        "| Suite / size | Cases | Old routes | New routes | Stock 2Q | Old Hybrid 2Q | "
        "New Hybrid 2Q | Commuting 2Q | New vs commuting | New wins/ties/losses |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for suite, n in (("full16", 16), ("scaling", 16), ("scaling", 20), ("scaling", 24)):
        group = [r for r in rows if r["suite"] == suite and int(r["logical_qubits"]) == n]
        if not group:
            continue
        stock = sum(int(r["stock_2q"]) for r in group)
        old = sum(int(r["old_hybrid_2q"]) for r in group)
        new = sum(int(r["dense_hybrid_2q"]) for r in group)
        commuting = sum(int(r["commuting_2q"]) for r in group)
        wins = sum(int(r["dense_hybrid_2q"]) < int(r["commuting_2q"]) for r in group)
        ties = sum(int(r["dense_hybrid_2q"]) == int(r["commuting_2q"]) for r in group)
        old_routes = sum(r["old_hybrid_status"] == "routed" for r in group)
        new_routes = sum(r["dense_hybrid_mode"].startswith("v2_rust") for r in group)
        lines.append(f"| {suite} {n}q | {len(group)} | {old_routes} | {new_routes} | "
                     f"{stock} | {old} | {new} | {commuting} | "
                     f"{(new / commuting - 1) * 100:+.1f}% | "
                     f"{wins}/{ties}/{len(group) - wins - ties} |")
    lines += ["", "## By graph family", "",
              "| Suite / size | Family | Cases | Old routes | New routes | Old Hybrid 2Q | "
              "New Hybrid 2Q | Commuting 2Q | New vs commuting |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for suite, n in (("full16", 16), ("scaling", 16), ("scaling", 20), ("scaling", 24)):
        families = sorted({r["family"] for r in rows
                           if r["suite"] == suite and int(r["logical_qubits"]) == n})
        for family in families:
            group = [r for r in rows if r["suite"] == suite and int(r["logical_qubits"]) == n
                     and r["family"] == family]
            old = sum(int(r["old_hybrid_2q"]) for r in group)
            new = sum(int(r["dense_hybrid_2q"]) for r in group)
            commuting = sum(int(r["commuting_2q"]) for r in group)
            lines.append(f"| {suite} {n}q | {family} | {len(group)} | "
                         f"{sum(r['old_hybrid_status'] == 'routed' for r in group)} | "
                         f"{sum(r['dense_hybrid_mode'].startswith('v2_rust') for r in group)} | "
                         f"{old} | {new} | {commuting} | "
                         f"{(new / commuting - 1) * 100:+.1f}% |")
    lines += ["", "## Compile time and selection", "",
              "| Suite / size | Median old Hybrid s | Median new Hybrid s | "
              "Median commuting s | Line / searched line / beam / existing / fallback |",
              "|---|---:|---:|---:|---:|"]
    for suite, n in (("full16", 16), ("scaling", 16), ("scaling", 20), ("scaling", 24)):
        group = [r for r in rows if r["suite"] == suite and int(r["logical_qubits"]) == n]
        if not group:
            continue
        count = {mode: sum(r["dense_hybrid_mode"] == mode for r in group)
                 for mode in ("v2_rust_dense_line", "v2_rust_dense_line_search",
                              "v2_rust_dense_beam")}
        existing = sum(r["dense_hybrid_mode"] == "v2_rust_layout_tournament" for r in group)
        fallback = len(group) - sum(count.values()) - existing
        lines.append(f"| {suite} {n}q | "
                     f"{median(float(r['old_hybrid_seconds']) for r in group):.3f} | "
                     f"{median(float(r['dense_hybrid_seconds']) for r in group):.3f} | "
                     f"{median(float(r['commuting_seconds']) for r in group):.3f} | "
                     f"{count['v2_rust_dense_line']}/{count['v2_rust_dense_line_search']}/"
                     f"{count['v2_rust_dense_beam']}/{existing}/{fallback} |")
    rss = [float(r["peak_rss_mib"]) for r in rows if r["peak_rss_mib"]]
    lines += ["", f"Peak process RSS: {max(rss):.1f} MiB." if rss else "Peak RSS unavailable.",
              f"Per-case data: {path.name}."]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_samples(output_name: str) -> Path:
    """Replay two dense routes and compare their compiled outputs with ideal QAOA."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator

    import hybrid_level3 as base
    try:
        from qaoa_hybrid import qaoa_v2_rust
    except ImportError:
        import qaoa_v2_rust
    from layout_seed import physical_line
    from research_scaling import mps_probability_error, route_replay
    from rust_router_core import optimise_line_layout_rust, route_line_layers_rust

    simulator = AerSimulator(method="matrix_product_state")
    lines = ["# Dense completion correctness checks", "",
             "Two selected 16-logical-qubit cases on full FakeBrisbane. Rust line routes "
             "were checked by replaying SWAPs and comparing the logical gate stream. "
             "The final native circuits were checked against ideal MaxCut probabilities.",
             "", "| Family | Layers | Selected mode | Native 2Q | "
             "Max probability error |", "|---|---:|---|---:|---:|"]
    checked = 0
    for suite, backend, qc, family, graph_id, layers, _ in cases(SimpleNamespace(suite="full16")):
        if backend.name != "fake_brisbane" or graph_id != 86000 \
                or (family, layers) not in (("dense75", 1), ("er75", 2)):
            continue
        path = physical_line(backend.coupling_map, 24)[:qc.num_qubits]
        order = optimise_line_layout_rust(qc, seed=11, max_seconds=0.15)
        for layout in (path, tuple(path[site] for site in order)):
            routed = route_line_layers_rust(qc.copy(), backend.coupling_map, path, layout)
            route_replay(qc, routed, layout, backend.coupling_map)
        logical = QuantumCircuit(qc.num_qubits, global_phase=qc.global_phase)
        for inst in qc.data:
            if isinstance(inst.operation, PauliEvolutionGate):
                for (u, v), angle in sorted(base._layer_terms(qc, inst).items()):
                    logical.rzz(float(angle), u, v)
            else:
                logical.append(inst.operation, [qc.find_bit(inst.qubits[0]).index])
        ideal = np.abs(Statevector.from_instruction(logical).data) ** 2
        compiled = compile_hybrid_dense_v2(qc.copy(), backend, seed_transpiler=11)
        error = mps_probability_error(compiled, ideal, simulator)
        twoq, _ = native_metrics(compiled, backend)
        lines.append(f"| {family} | {layers} | {compiled.route_mode} | "
                     f"{twoq} | {error:.3e} |")
        checked += 1
        print(f"{family} p={layers}: symbolic replay and compiled MPS PASS "
              f"({error:.3e})", flush=True)
    if checked != 2:
        raise AssertionError(f"Expected two correctness samples, found {checked}")
    scaling_checks = 0
    for suite, backend, qc, family, graph_id, layers, _ in cases(SimpleNamespace(suite="scaling")):
        if backend.name != "fake_brisbane" or family != "dense75" or graph_id != 94000:
            continue
        path = physical_line(backend.coupling_map, 24)[:qc.num_qubits]
        order = optimise_line_layout_rust(qc, seed=11, max_seconds=0.15)
        layout = tuple(path[site] for site in order)
        routed = route_line_layers_rust(qc.copy(), backend.coupling_map, path, layout)
        route_replay(qc, routed, layout, backend.coupling_map)
        scaling_checks += 1
    if scaling_checks != 6:
        raise AssertionError(f"Expected six scaling route checks, found {scaling_checks}")
    for n in range(2, 25):
        pairs = [(u, v, 0.1) for u in range(n) for v in range(u + 1, n)]
        events, positions, _ = qaoa_v2_rust.route_line_layer(
            pairs, list(range(n)), n, [(q, q + 1) for q in range(n - 1)],
            list(range(n)))
        if sum(event[0] == "rzz" for event in events) != len(pairs) \
                or set(positions) != set(range(n)):
            raise AssertionError(f"Complete-graph line route failed at {n} qubits")
    lines += ["", "Six more dense routes at 16, 20, and 24 logical qubits "
              "(one and two layers each) passed symbolic SWAP and gate-stream replay. "
              "Complete-graph line scheduling also passed for every size from 2 to 24. "
              "Probability agreement checks the measured distribution, not global phase. "
              "The 24-qubit cases were not simulated with a 2^24 statevector."]
    path = ROOT / "results" / f"{output_name}_checks.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "full16", "scaling"), default="all")
    parser.add_argument("--backends", default="fake_brisbane,fake_kyiv,fake_sherbrooke")
    parser.add_argument("--graph-id", type=int)
    parser.add_argument("--layers", type=int, choices=(1, 2))
    parser.add_argument("--output-name", default="v2_dense_rescue_348")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    if args.output_name != Path(args.output_name).name or not args.output_name.startswith("v2_"):
        parser.error("Output name must be a plain v2_* prefix")
    selected_backends = set(args.backends.split(","))
    if not selected_backends <= {"fake_brisbane", "fake_kyiv", "fake_sherbrooke"}:
        parser.error("Unknown backend")
    if args.checks_only:
        print(verify_samples(args.output_name))
        return
    output = ROOT / "results" / f"{args.output_name}.csv"
    if output.exists() and not args.resume:
        raise FileExistsError(f"Would overwrite existing result: {output}")
    rows = (list(csv.DictReader(output.open(newline="", encoding="utf-8")))
            if output.exists() else [])
    seen = {key(row) for row in rows}
    if len(seen) != len(rows):
        raise AssertionError("Duplicate rows in saved result")
    old = references()
    selected = 0
    for suite, backend, qc, family, graph_id, layers, edge_count in cases(args):
        identity = (suite, backend.name, qc.num_qubits, family, graph_id, layers)
        if backend.name not in selected_backends or args.graph_id not in (None, graph_id) \
                or args.layers not in (None, layers):
            continue
        selected += 1
        if identity in seen:
            continue
        reference = old[identity]
        started = perf_counter()
        compiled = compile_hybrid_dense_v2(qc.copy(), backend, seed_transpiler=11)
        elapsed = perf_counter() - started
        twoq, depth = native_metrics(compiled, backend)
        _, peak = process_memory_mib()
        row = dict(
            suite=suite, snapshot=backend.name, logical_qubits=qc.num_qubits,
            family=family, graph_id=graph_id, layers=layers, edges=edge_count,
            stock_2q=reference["stock_2q"],
            old_hybrid_2q=reference["hybrid_2q"], dense_hybrid_2q=twoq,
            commuting_2q=reference["commuting_2q"],
            stock_2q_depth=reference["stock_2q_depth"],
            old_hybrid_2q_depth=reference["hybrid_2q_depth"],
            dense_hybrid_2q_depth=depth,
            commuting_2q_depth=reference["commuting_2q_depth"],
            old_hybrid_seconds=reference["hybrid_seconds"],
            dense_hybrid_seconds=elapsed, commuting_seconds=reference["commuting_seconds"],
            old_hybrid_status=reference["hybrid_status"],
            dense_hybrid_mode=compiled.route_mode,
            peak_rss_mib=peak if peak is not None else "",
        )
        rows.append(row)
        seen.add(identity)
        save(output, rows)
        report(output, rows)
        print(f"[{len(rows)}/{selected}] {backend.name} {suite} {qc.num_qubits}q "
              f"{family} graph={graph_id} p={layers}: {twoq} 2Q "
              f"({compiled.route_mode}, {elapsed:.2f}s)", flush=True)
    if not selected:
        raise ValueError("No cases matched the requested filters")
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
