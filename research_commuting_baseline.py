"""Paired Stock, Hybrid v2, and Qiskit commuting-router + SAT line benchmark."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from itertools import combinations
from pathlib import Path
from statistics import median
from threading import Timer
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from pysat.solvers import Solver
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    Commuting2qGateRouter, FindCommutingPauliEvolutions, SwapStrategy,
)
from qiskit_aer import AerSimulator

import hybrid_level3 as base
from hybrid_level3_v2 import compile_hybrid_level3_v2
from layout_seed import physical_line
from research_no_reroute import cases
from research_scaling import mps_probability_error, native_metrics, process_memory_mib


ROOT = Path(__file__).resolve().parent
FIELDS = ("suite", "snapshot", "logical_qubits", "family", "graph_id", "layers",
          "edges", "stock_2q", "hybrid_2q", "commuting_2q",
          "stock_2q_depth", "hybrid_2q_depth", "commuting_2q_depth",
          "stock_depth", "hybrid_depth", "commuting_depth",
          "stock_seconds", "hybrid_seconds", "commuting_seconds", "sat_seconds",
          "hybrid_status", "commuting_status", "sat_layers", "swap_layers", "sat_optimal",
          "mps_stock_error", "mps_hybrid_error", "mps_commuting_error", "peak_rss_mib")


def sat_line_mapping(n: int, edges: tuple[tuple[int, int], ...],
                     seconds_per_check: float) -> tuple[tuple[int, ...], int, bool, float]:
    """Map graph vertices to line sites with the SAT connectivity constraints."""
    started = perf_counter()
    distance = SwapStrategy.from_line(list(range(n))).distance_matrix
    variable = lambda u, p: u * n + p + 1
    one_to_one = []
    for u in range(n):
        one_to_one.append([variable(u, p) for p in range(n)])
        one_to_one.extend([-variable(u, p), -variable(u, q)]
                          for p, q in combinations(range(n), 2))
    for p in range(n):
        one_to_one.extend([-variable(u, p), -variable(v, p)]
                          for u, v in combinations(range(n), 2))
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    low = max(0, max(degree) - 2)
    high = n - 2
    best = tuple(range(n))
    timed_out = False
    while low < high:
        layer_limit = (low + high) // 2
        clauses = one_to_one.copy()
        for u, v in edges:
            for p in range(n):
                neighbor_sites = [variable(v, q) for q in range(n)
                                  if p != q and 0 <= distance[p, q] <= layer_limit]
                clauses.append([-variable(u, p), *neighbor_sites])
        with Solver(bootstrap_with=clauses) as solver:
            timer = Timer(seconds_per_check, solver.interrupt)
            timer.start()
            try:
                status = solver.solve_limited(expect_interrupt=True)
                model = solver.get_model() if status else None
            finally:
                timer.cancel()
        if status is None:
            timed_out = True
            break
        if status:
            mapping = [None] * n
            for literal in model:
                if 0 < literal <= n * n:
                    logical, position = divmod(literal - 1, n)
                    mapping[logical] = position
            if set(mapping) != set(range(n)):
                raise AssertionError("SAT returned an incomplete line mapping")
            best = tuple(mapping)
            high = layer_limit
        else:
            low = layer_limit + 1
    if any(distance[best[u], best[v]] > high for u, v in edges):
        raise AssertionError("SAT line mapping does not cover every ZZ interaction")
    return best, high, not timed_out, perf_counter() - started


def verify_commuting_route(qc, routed, permutation: tuple[int, ...],
                           expected_positions: tuple[int, ...]) -> tuple[int, ...]:
    """Replay SWAPs and check every logical qubit's QAOA gate dependencies."""
    n = qc.num_qubits
    expected = [[Counter() for _ in range(sum(isinstance(inst.operation, PauliEvolutionGate)
                                                 for inst in qc.data))] for _ in range(n)]
    rx_expected = [[] for _ in range(n)]
    layer = 0
    for inst in qc.data:
        if isinstance(inst.operation, PauliEvolutionGate):
            for (u, v), angle in base._layer_terms(qc, inst).items():
                angle = round(float(angle), 10)
                expected[u][layer][v, angle] += 1
                expected[v][layer][u, angle] += 1
            layer += 1
        elif inst.operation.name == "rx":
            q = qc.find_bit(inst.qubits[0]).index
            rx_expected[q].append(round(float(inst.operation.params[0]), 10))
    at = list(range(n))
    h_seen = [False] * n
    rx_seen = [0] * n
    actual = [[Counter() for _ in range(layer)] for _ in range(n)]
    for inst in routed.data:
        gate = inst.operation
        sites = tuple(routed.find_bit(q).index for q in inst.qubits)
        if gate.name == "swap":
            a, b = sites
            if abs(permutation[a] - permutation[b]) != 1:
                raise AssertionError("Commuting router swapped nonadjacent line sites")
            at[a], at[b] = at[b], at[a]
        elif gate.name == "h":
            logical = at[sites[0]]
            if h_seen[logical] or rx_seen[logical]:
                raise AssertionError("Unexpected H gate order")
            h_seen[logical] = True
        elif gate.name == "rx":
            logical = at[sites[0]]
            index = rx_seen[logical]
            if index >= layer or round(float(gate.params[0]), 10) != rx_expected[logical][index]:
                raise AssertionError("Unexpected mixer gate")
            rx_seen[logical] += 1
        elif isinstance(gate, PauliEvolutionGate):
            if len(sites) != 2 or abs(permutation[sites[0]] - permutation[sites[1]]) != 1:
                raise AssertionError("Commuting gate is not on a line edge")
            terms = gate.operator.to_sparse_list()
            if len(terms) != 1 or terms[0][0] != "ZZ":
                raise AssertionError("Unexpected commuting evolution term")
            coeff = complex(terms[0][2])
            if abs(coeff.imag) > 1e-12:
                raise AssertionError("Complex ZZ coefficient")
            u, v = (at[site] for site in sites)
            step = rx_seen[u]
            if step != rx_seen[v] or step >= layer or not h_seen[u] or not h_seen[v]:
                raise AssertionError("ZZ gate crossed a logical layer boundary")
            angle = round(2 * float(gate.time) * coeff.real, 10)
            actual[u][step][v, angle] += 1
            actual[v][step][u, angle] += 1
        else:
            raise AssertionError(f"Unexpected commuting output gate: {gate.name}")
    if not all(h_seen) or rx_seen != [layer] * n:
        raise AssertionError("Commuting route changed the H/RX gate stream")
    if actual != expected:
        bad = next(q for q in range(n) if actual[q] != expected[q])
        step = next(i for i in range(layer) if actual[bad][i] != expected[bad][i])
        raise AssertionError(f"Commuting ZZ stream differs on logical {bad}, layer {step}: "
                             f"missing={list((expected[bad][step] - actual[bad][step]).items())[:3]} "
                             f"extra={list((actual[bad][step] - expected[bad][step]).items())[:3]}")
    positions = tuple(at.index(q) for q in range(n))
    if positions != expected_positions:
        raise AssertionError("Commuting final positions disagree with layer routing")
    return positions


def compile_commuting(qc, backend, edges, path, seed: int, sat_seconds: float):
    started = perf_counter()
    n = qc.num_qubits
    permutation, layers, optimal, sat_time = sat_line_mapping(n, edges, sat_seconds)
    line = [None] * n
    for logical, position in enumerate(permutation):
        line[position] = logical
    pre = QuantumCircuit(n, global_phase=qc.global_phase)
    logical_positions = list(range(n))
    cost_index = 0
    for inst in qc.data:
        if isinstance(inst.operation, PauliEvolutionGate):
            cost = QuantumCircuit(n)
            mapped_terms = [(pauli, [logical_positions[q] for q in sites], coefficient)
                            for pauli, sites, coefficient in
                            inst.operation.operator.to_sparse_list()]
            mapped_operator = SparsePauliOp.from_sparse_list(mapped_terms, num_qubits=n)
            cost.append(PauliEvolutionGate(mapped_operator, time=inst.operation.time),
                        cost.qubits)
            swap_strategy = SwapStrategy.from_line(
                line, num_swap_layers=layers if cost_index == 0 else n - 2)
            routed_layer = PassManager([
                FindCommutingPauliEvolutions(), Commuting2qGateRouter(swap_strategy),
            ]).run(cost)
            for routed_inst in routed_layer.data:
                sites = [routed_layer.find_bit(q).index for q in routed_inst.qubits]
                pre.append(routed_inst.operation, [pre.qubits[site] for site in sites])
            pre.global_phase += routed_layer.global_phase
            after = routed_layer.layout.final_index_layout()[:n]
            logical_positions = [after[logical_positions[q]] for q in range(n)]
            cost_index += 1
        else:
            logical = qc.find_bit(inst.qubits[0]).index
            pre.append(inst.operation, [pre.qubits[logical_positions[logical]]])
    pre_positions = verify_commuting_route(qc, pre, permutation,
                                           tuple(logical_positions))
    manager = base._build_level3(
        backend.target,
        initial_layout=[path[permutation[q]] for q in range(n)],
        seed_transpiler=seed, routing_method="none")
    native = manager.run(pre)
    physical_positions = base._final_positions(native, n)
    positions = [physical_positions[pre_positions[q]] for q in range(n)]
    result = SimpleNamespace(circuit=native, final_positions=positions)
    count, depth = native_metrics(result, backend)
    return result, count, depth, perf_counter() - started, sat_time, layers, n - 2 if cost_index > 1 else layers, optimal


def save(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def report(path: Path, rows: list[dict], args) -> None:
    snapshots = ", ".join(sorted({row["snapshot"] for row in rows}))
    lines = ["# QAOA commuting-router baseline on IBM fake targets", "",
             f"Paired local compilation with seed 11 on full targets: {snapshots}. "
             "All three methods see the same MaxCut circuit. "
             "Qiskit's specialized baseline uses FindCommutingPauliEvolutions, "
             "Commuting2qGateRouter, a fixed physical line, a SAT line mapping, and final "
             "Level-3 native compilation without rerouting. SAT time is included. "
             "Each ZZ layer is routed in turn while carrying its final logical placement "
             "into the next layer. Later layers use the complete line strategy. "
             "Hybrid is v2.0.0 with quality_guard=False, so a Rust timeout is marked "
             "and falls back to Stock. This is not a QPU measurement.", "",
             f"SAT limit: {args.sat_seconds:g} s per feasibility check. "
             "A timeout keeps the best feasible mapping found so far; it does not claim optimality.",
             "", "| Suite / size | Cases | Rust routes | First-layer SAT proven | Stock 2Q | "
             "Hybrid 2Q | Commuting 2Q | Hybrid vs Stock | Commuting vs Stock | "
             "Hybrid vs commuting |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for suite, n in (("full16", 16), ("scaling", 16), ("scaling", 20), ("scaling", 24)):
        group = [row for row in rows if row["suite"] == suite and int(row["logical_qubits"]) == n]
        if not group:
            continue
        complete = [row for row in group if row["commuting_status"] == "compiled"]
        stock = sum(int(row["stock_2q"]) for row in complete)
        hybrid = sum(int(row["hybrid_2q"]) for row in complete)
        commuting = sum(int(row["commuting_2q"]) for row in complete)
        lines.append(f"| {suite} {n}q | {len(complete)}/{len(group)} | "
                     f"{sum(row['hybrid_status'] == 'routed' for row in complete)} | "
                     f"{sum(str(row['sat_optimal']) == 'True' for row in complete)} | "
                     f"{stock} | {hybrid} | {commuting} | "
                     f"{(hybrid / stock - 1) * 100:+.1f}% | "
                     f"{(commuting / stock - 1) * 100:+.1f}% | "
                     f"{(hybrid / commuting - 1) * 100:+.1f}% |")
    lines += ["", "## Time and direct comparison", "",
              "All medians use completed commuting rows. Hybrid ties caused by Stock "
              "fallback remain in the totals above.", "",
              "| Suite / size | Hybrid better/tie/worse vs commuting | "
              "Median Stock s | Median Hybrid s | Median commuting s |",
              "|---|---:|---:|---:|---:|"]
    for suite, n in (("full16", 16), ("scaling", 16), ("scaling", 20), ("scaling", 24)):
        group = [row for row in rows if row["suite"] == suite and int(row["logical_qubits"]) == n
                 and row["commuting_status"] == "compiled"]
        if not group:
            continue
        h = [int(row["hybrid_2q"]) for row in group]
        c = [int(row["commuting_2q"]) for row in group]
        wins = sum(a < b for a, b in zip(h, c))
        ties = sum(a == b for a, b in zip(h, c))
        lines.append(f"| {suite} {n}q | {wins}/{ties}/{len(group) - wins - ties} | "
                     f"{median(float(row['stock_seconds']) for row in group):.3f} | "
                     f"{median(float(row['hybrid_seconds']) for row in group):.3f} | "
                     f"{median(float(row['commuting_seconds']) for row in group):.3f} |")
    lines += ["", "## By graph family", "",
              "A positive Hybrid-versus-commuting percentage means Hybrid used more native 2Q gates.",
              "", "| Suite / size | Family | Cases | Rust routes | Stock 2Q | Hybrid 2Q | "
              "Commuting 2Q | Hybrid vs commuting |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for suite, n in (("full16", 16), ("scaling", 16), ("scaling", 20), ("scaling", 24)):
        families = sorted({row["family"] for row in rows if row["suite"] == suite
                           and int(row["logical_qubits"]) == n})
        for family in families:
            group = [row for row in rows if row["suite"] == suite
                     and int(row["logical_qubits"]) == n and row["family"] == family
                     and row["commuting_status"] == "compiled"]
            if not group:
                continue
            stock = sum(int(row["stock_2q"]) for row in group)
            hybrid = sum(int(row["hybrid_2q"]) for row in group)
            commuting = sum(int(row["commuting_2q"]) for row in group)
            routed = sum(row["hybrid_status"] == "routed" for row in group)
            lines.append(f"| {suite} {n}q | {family} | {len(group)} | {routed} | "
                         f"{stock} | {hybrid} | {commuting} | "
                         f"{(hybrid / commuting - 1) * 100:+.1f}% |")
    lines += ["", "## Cases where Rust finished", "",
              "Only matching rows with a completed Rust route are counted here.", "",
              "| Suite / size | Cases | Stock 2Q | Hybrid 2Q | Commuting 2Q | "
              "Hybrid vs commuting | Hybrid wins/ties/losses |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for suite, n in (("full16", 16), ("scaling", 16), ("scaling", 20), ("scaling", 24)):
        group = [row for row in rows if row["suite"] == suite
                 and int(row["logical_qubits"]) == n and row["hybrid_status"] == "routed"
                 and row["commuting_status"] == "compiled"]
        if not group:
            continue
        stock = sum(int(row["stock_2q"]) for row in group)
        hybrid = sum(int(row["hybrid_2q"]) for row in group)
        commuting = sum(int(row["commuting_2q"]) for row in group)
        wins = sum(int(row["hybrid_2q"]) < int(row["commuting_2q"]) for row in group)
        ties = sum(int(row["hybrid_2q"]) == int(row["commuting_2q"]) for row in group)
        lines.append(f"| {suite} {n}q | {len(group)} | {stock} | {hybrid} | "
                     f"{commuting} | {(hybrid / commuting - 1) * 100:+.1f}% | "
                     f"{wins}/{ties}/{len(group) - wins - ties} |")
    failures = [row for row in rows if row["commuting_status"] != "compiled"]
    checks = [float(row[name]) for row in rows for name in
              ("mps_stock_error", "mps_hybrid_error", "mps_commuting_error")
              if row[name] not in ("", None)]
    rss = [float(row["peak_rss_mib"]) for row in rows if row["peak_rss_mib"] not in ("", None)]
    lines += ["", f"Commuting compilation failures: {len(failures)}.",
              f"Compiled MPS probability checks: {len(checks)}; max error "
              f"{max(checks) if checks else 'n/a'}.",
              f"Peak process RSS: {max(rss):.1f} MiB." if rss else "Peak RSS unavailable.",
              "Every commuting route has exact symbolic gate and SWAP replay checks. "
              "MPS probabilities do not prove phase equivalence.",
              f"Per-case data: {path.name}."]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "full16", "scaling"), default="scaling")
    parser.add_argument("--backends", default="fake_brisbane,fake_kyiv,fake_sherbrooke")
    parser.add_argument("--graph-id", type=int)
    parser.add_argument("--layers", type=int, choices=(1, 2))
    parser.add_argument("--sat-seconds", type=float, default=2.0)
    parser.add_argument("--mps16", action="store_true")
    parser.add_argument("--output-name", default="v2_commuting_sat_scaling")
    args = parser.parse_args()
    if args.sat_seconds <= 0 or args.output_name != Path(args.output_name).name \
            or not args.output_name.startswith("v2_"):
        parser.error("SAT limit must be positive; output name must be a plain v2_* prefix")
    selected_backends = set(args.backends.split(","))
    if not selected_backends <= {"fake_brisbane", "fake_kyiv", "fake_sherbrooke"}:
        parser.error("Unknown backend")
    path = ROOT / "results" / f"{args.output_name}.csv"
    if path.exists():
        raise FileExistsError(f"Would overwrite existing results: {path}")
    rows = []
    simulator = AerSimulator(method="matrix_product_state") if args.mps16 else None
    backend_paths = {}
    for suite, backend, qc, family, graph_id, layers, edge_count in cases(args):
        if backend.name not in selected_backends or args.graph_id not in (None, graph_id) \
                or args.layers not in (None, layers):
            continue
        if backend.name not in backend_paths:
            backend_paths[backend.name] = physical_line(backend.coupling_map, 24)
        n = qc.num_qubits
        first_layer = next(inst for inst in qc.data if isinstance(inst.operation, PauliEvolutionGate))
        edges = tuple(sorted(base._layer_terms(qc, first_layer)))
        print(f"[{len(rows) + 1}] {backend.name} {suite} {n}q {family} "
              f"graph={graph_id} p={layers}", flush=True)
        start = perf_counter()
        stock = base.compile_stock_level3(qc.copy(), backend, seed_transpiler=11)
        stock_seconds = perf_counter() - start
        start = perf_counter()
        hybrid = compile_hybrid_level3_v2(
            qc.copy(), backend, seed_transpiler=11, layouts=1, finalists=4,
            beam_seconds=1.0, quality_guard=False)
        hybrid_seconds = perf_counter() - start
        stock_2q, stock_depth = native_metrics(stock, backend)
        hybrid_2q, hybrid_depth = native_metrics(hybrid, backend)
        commuting, commuting_2q, commuting_depth, commuting_seconds, sat_seconds, sat_layers, swap_layers, sat_optimal = (
            compile_commuting(qc, backend, edges, backend_paths[backend.name][:n], 11,
                              args.sat_seconds))
        row = dict(suite=suite, snapshot=backend.name, logical_qubits=n, family=family,
                   graph_id=graph_id, layers=layers, edges=edge_count,
                   stock_2q=stock_2q, hybrid_2q=hybrid_2q, commuting_2q=commuting_2q,
                   stock_2q_depth=stock_depth, hybrid_2q_depth=hybrid_depth,
                   commuting_2q_depth=commuting_depth,
                   stock_depth=stock.circuit.depth(), hybrid_depth=hybrid.circuit.depth(),
                   commuting_depth=commuting.circuit.depth(), stock_seconds=stock_seconds,
                   hybrid_seconds=hybrid_seconds, commuting_seconds=commuting_seconds,
                   sat_seconds=sat_seconds,
                   hybrid_status=("routed" if hybrid.route_mode == "v2_rust_layout_tournament"
                                  else "stock_fallback"),
                   commuting_status="compiled", sat_layers=sat_layers,
                   swap_layers=swap_layers, sat_optimal=sat_optimal,
                   mps_stock_error="", mps_hybrid_error="", mps_commuting_error="",
                   peak_rss_mib="")
        if simulator is not None and n == 16 and graph_id in (86000, 94000) \
                and backend.name == "fake_brisbane":
            logical = QuantumCircuit(n, global_phase=qc.global_phase)
            for inst in qc.data:
                if isinstance(inst.operation, PauliEvolutionGate):
                    for (u, v), angle in sorted(base._layer_terms(qc, inst).items()):
                        logical.rzz(float(angle), u, v)
                else:
                    logical.append(inst.operation,
                                   [qc.find_bit(inst.qubits[0]).index])
            ideal = np.abs(Statevector.from_instruction(logical).data) ** 2
            for name, compiled in (("stock", stock), ("hybrid", hybrid),
                                   ("commuting", commuting)):
                row[f"mps_{name}_error"] = mps_probability_error(compiled, ideal, simulator)
        _, peak = process_memory_mib()
        row["peak_rss_mib"] = peak if peak is not None else ""
        rows.append(row)
        save(path, rows)
        report(path, rows, args)
    if not rows:
        raise ValueError("No cases matched the requested filters")
    print(path.with_suffix(".md"))


if __name__ == "__main__":
    main()
