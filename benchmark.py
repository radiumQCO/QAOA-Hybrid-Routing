"""One-command benchmark for stock Qiskit Level 3 vs my Hybrid Level 3.

Every profile runs both suites automatically:
  1) balanced density/structure coverage,
  2) a structure-diverse QAOA workload suite.

The standard and stress profiles also run local hardware-calibrated noise simulation
on a subset. No IBM account is needed. This is still a FakeGuadalupeV2 snapshot,
not a claim about a current physical IBM QPU.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import time
import warnings

import numpy as np
from scipy.sparse import SparseEfficiencyWarning
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector

try:
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
except ImportError as exc:
    raise SystemExit(
        "Missing benchmark packages. Run: python -m pip install -r requirements.txt"
    ) from exc

from hybrid_level3 import compile_hybrid_level3, compile_stock_level3


N = 16
BASE_SEED = 20260924
STATE_TOL = 1e-4
TV_TOL = 1e-4
CUT_TOL = 1e-3

# I keep three sizes because I do not want every useful test to become an all-night job.
# `standard` is the one I actually expect to use most of the time. On the PC where
# quick took about 75 seconds, I am aiming for roughly 40-90 minutes here.
PROFILES = {
    "quick": {
        "balanced_instances": 1,
        "workload_instances": 1,
        "compiler_seeds": (11,),
        "cases": (1, 2),
        "shots": 64,
        "bootstrap_reps": 1000,
        "noise_instances": 1,
        "beam_seconds": 5.0,
    },
    "standard": {
        "balanced_instances": 12,
        "workload_instances": 12,
        "compiler_seeds": (11, 29, 47),
        "cases": (1, 2),
        "shots": 128,
        "bootstrap_reps": 5000,
        "noise_instances": 1,
        "beam_seconds": 5.0,
    },
    "stress": {
        "balanced_instances": 20,
        "workload_instances": 20,
        "compiler_seeds": (11, 29, 47, 71, 97),
        "cases": (1, 2),
        "shots": 512,
        "bootstrap_reps": 10000,
        "noise_instances": 4,
        "beam_seconds": 5.0,
    },
}

BALANCED_FAMILIES = (
    "very_sparse15",
    "sparse25",
    "medium50",
    "dense75",
    "nearcomplete90",
    "ring",
    "all_to_all",
)

WORKLOAD_FAMILIES = (
    "cubic",
    "quartic",
    "grid4x4",
    "community",
    "er25",
    "er50",
    "er75",
)

RAW_FIELDS = (
    "suite",
    "family",
    "graph_id",
    "graph_seed",
    "density",
    "case",
    "layers",
    "method",
    "compiler_seed",
    "route_mode",
    "selector_reason",
    "native_2q",
    "cx",
    "depth",
    "twoq_depth",
    "duration_us",
    "router_seconds",
    "level3_seconds",
    "total_compile_seconds",
    "swaps_before_level3",
    "state_error",
    "total_variation",
    "ideal_cut",
    "compiled_ideal_cut",
    "noise_simulated",
    "shots",
    "noisy_cut",
    "noisy_minus_ideal",
    "simulator_seconds",
    "final_positions",
)

SUMMARY_FIELDS = (
    "scope",
    "suite",
    "family",
    "case",
    "pairs",
    "native_2q_saving_pct",
    "native_2q_ci_low",
    "native_2q_ci_high",
    "duration_saving_pct",
    "twoq_depth_saving_pct",
    "quantum_depth_saving_pct",
    "compile_time_change_pct",
    "wins",
    "ties",
    "losses",
    "noisy_cut_delta",
    "noisy_pairs",
)


@dataclass(frozen=True)
class GraphSpec:
    suite: str
    family: str
    graph_id: int
    graph_seed: int
    edges: tuple[tuple[int, int], ...]

    @property
    def density(self) -> float:
        return len(self.edges) / (N * (N - 1) / 2)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def all_edges():
    return [(a, b) for a in range(N) for b in range(a + 1, N)]


def random_density_edges(density: float, seed: int):
    possible = all_edges()
    rng = np.random.default_rng(seed)
    count = max(1, min(len(possible), int(round(len(possible) * density))))
    picks = rng.choice(len(possible), count, replace=False)
    return tuple(sorted(possible[int(i)] for i in picks))


def ring_edges():
    return tuple((q, q + 1) for q in range(N - 1)) + ((0, N - 1),)


def grid_edges():
    edges = []
    side = 4
    for row in range(side):
        for col in range(side):
            q = row * side + col
            if row + 1 < side:
                edges.append((q, q + side))
            if col + 1 < side:
                edges.append((q, q + 1))
    return tuple(sorted(edges))


def _extra_matching(existing: set[tuple[int, int]], rng) -> set[tuple[int, int]]:
    nodes = np.arange(N)
    for _ in range(5000):
        rng.shuffle(nodes)
        proposed = set()
        good = True
        for i in range(0, N, 2):
            edge = tuple(sorted((int(nodes[i]), int(nodes[i + 1]))))
            if edge in existing or edge in proposed:
                good = False
                break
            proposed.add(edge)
        if good:
            return proposed
    raise RuntimeError("Could not build a clean random matching.")


def regular_edges(degree: int, seed: int):
    if degree not in (3, 4):
        raise ValueError("This small generator only supports degree 3 or 4.")
    rng = np.random.default_rng(seed)
    edges = set(ring_edges())
    while any(sum(q in edge for edge in edges) < degree for q in range(N)):
        edges.update(_extra_matching(edges, rng))
    degrees = {q: sum(q in edge for edge in edges) for q in range(N)}
    if any(value != degree for value in degrees.values()):
        raise RuntimeError("Regular-graph generator produced the wrong degree.")
    return tuple(sorted(edges))


def community_edges(seed: int):
    rng = np.random.default_rng(seed)
    edges = set()
    for a in range(N):
        for b in range(a + 1, N):
            same = (a < 8) == (b < 8)
            probability = 0.55 if same else 0.08
            if rng.random() < probability:
                edges.add((a, b))
    edges.add((3, 12))
    for q in range(N):
        if not any(q in edge for edge in edges):
            mate = (q + 1) % 8 if q < 8 else 8 + ((q - 8 + 1) % 8)
            edges.add(tuple(sorted((q, mate))))
    return tuple(sorted(edges))


def family_edges(suite: str, family: str, seed: int):
    if suite == "balanced":
        mapping = {
            "very_sparse15": 0.15,
            "sparse25": 0.25,
            "medium50": 0.50,
            "dense75": 0.75,
            "nearcomplete90": 0.90,
        }
        if family in mapping:
            return random_density_edges(mapping[family], seed)
        if family == "ring":
            return ring_edges()
        if family == "all_to_all":
            return tuple(all_edges())
    elif suite == "workload":
        if family == "cubic":
            return regular_edges(3, seed)
        if family == "quartic":
            return regular_edges(4, seed)
        if family == "grid4x4":
            return grid_edges()
        if family == "community":
            return community_edges(seed)
        if family == "er25":
            return random_density_edges(0.25, seed)
        if family == "er50":
            return random_density_edges(0.50, seed)
        if family == "er75":
            return random_density_edges(0.75, seed)
    raise ValueError(f"Unknown suite/family: {suite}/{family}")


def graph_specs(profile_name: str, graph_start: int):
    # Deterministic shapes only need one copy. Random families get many fresh seeds,
    # otherwise I would just be benchmarking the same graph again and again.
    profile = PROFILES[profile_name]
    specs = []
    suite_data = (
        ("balanced", BALANCED_FAMILIES, profile["balanced_instances"]),
        ("workload", WORKLOAD_FAMILIES, profile["workload_instances"]),
    )
    for suite_index, (suite, families, random_instances) in enumerate(suite_data):
        for family_index, family in enumerate(families):
            deterministic = family in {"ring", "all_to_all", "grid4x4"}
            count = 1 if deterministic else random_instances
            for offset in range(count):
                graph_id = graph_start + offset
                graph_seed = (
                    BASE_SEED
                    + suite_index * 10_000_000
                    + family_index * 1_000_000
                    + graph_id
                )
                specs.append(
                    GraphSpec(
                        suite,
                        family,
                        graph_id,
                        graph_seed,
                        family_edges(suite, family, graph_seed),
                    )
                )
    return specs


def qaoa_circuit(edges, layers: int, angle_seed: int):
    rng = np.random.default_rng(angle_seed)
    angles = rng.uniform(0.15, 0.95, size=(layers, 2))
    circuit = QuantumCircuit(N)
    circuit.h(range(N))
    operator = SparsePauliOp.from_sparse_list(
        [("ZZ", [a, b], 1.0) for a, b in edges],
        num_qubits=N,
    )
    for gamma, beta in angles:
        circuit.append(PauliEvolutionGate(operator, time=float(gamma)), range(N))
        circuit.rx(2 * float(beta), range(N))
    return circuit, angles.ravel()


def reference_statevector(edges, flat_angles):
    """Build the ideal QAOA state without materializing a 2**N by 2**N matrix.

    Statevector.from_instruction(PauliEvolutionGate(...)) can ask Qiskit/SciPy
    to turn a 16-qubit evolution into a dense 65536x65536 operator (~64 GiB).
    All ZZ terms commute, so the same evolution is exactly a sequence of
    RZZ(2*gamma) gates, which the statevector simulator applies using small
    two-qubit operations and only needs the 2**N state vector.
    """
    # This exists because the first version accidentally tried to allocate ~64 GiB.
    # RZZ gives me the same ideal state here without doing something ridiculous.
    values = np.asarray(flat_angles, dtype=float).reshape(-1, 2)
    circuit = QuantumCircuit(N)
    circuit.h(range(N))
    for gamma, beta in values:
        for a, b in edges:
            circuit.rzz(2.0 * float(gamma), a, b)
        circuit.rx(2.0 * float(beta), range(N))
    return Statevector.from_instruction(circuit).data


def cut_scores(edges):
    states = np.arange(1 << N, dtype=np.uint32)
    scores = np.zeros(states.size, dtype=np.uint8)
    for a, b in edges:
        scores += ((states >> a) ^ (states >> b)) & 1
    return scores


def amplitude_error(reference, candidate):
    overlap = np.vdot(reference, candidate)
    aligned = candidate * np.exp(-1j * np.angle(overlap))
    return float(np.linalg.norm(aligned - reference))


def logical_amplitudes(circuit, positions):
    physical = Statevector.from_instruction(circuit).data
    logical_indices = np.arange(1 << N, dtype=np.uint32)
    physical_indices = np.zeros(logical_indices.size, dtype=np.uint32)
    for logical, site in enumerate(positions):
        physical_indices |= ((logical_indices >> logical) & 1) << int(site)
    return physical[physical_indices]


def twoq_depth(circuit):
    levels = [0] * circuit.num_qubits
    maximum = 0
    for instruction in circuit.data:
        if instruction.operation.num_qubits != 2:
            continue
        qids = [circuit.find_bit(q).index for q in instruction.qubits]
        level = max(levels[q] for q in qids) + 1
        for q in qids:
            levels[q] = level
        maximum = max(maximum, level)
    return maximum


def circuit_metrics(circuit, target):
    counts = circuit.count_ops()
    native_2q = sum(
        1
        for instruction in circuit.data
        if instruction.operation.num_qubits == 2 and instruction.operation.name != "barrier"
    )
    return {
        "native_2q": int(native_2q),
        "cx": int(counts.get("cx", 0)),
        "depth": int(circuit.depth()),
        "twoq_depth": int(twoq_depth(circuit)),
        "duration_us": float(circuit.estimate_duration(target, unit="u")),
    }


def measured_circuit(compiled, positions):
    out = compiled.copy()
    bits = ClassicalRegister(N, "logical")
    out.add_register(bits)
    for logical, physical in enumerate(positions):
        out.measure(int(physical), bits[logical])
    return out


def shot_mean(counts, scores):
    shots = sum(counts.values())
    total = 0.0
    for bits, count in counts.items():
        logical_bits = bits.replace(" ", "")
        total += float(scores[int(logical_bits, 2)]) * count
    return total / shots


def should_simulate_noise(profile_name, suite, family, instance_index, layers):
    # Noise simulation is useful, but it is also one of the slowest parts of the test.
    # I sample it instead of turning a compiler benchmark into a simulator benchmark.
    if profile_name == "quick":
        return layers == 1 and instance_index == 0 and (
            (suite == "balanced" and family == "medium50")
            or (suite == "workload" and family == "cubic")
        )
    if profile_name == "standard":
        # Two V4-ish cases and two Qiskit-fallback cases is enough to catch obvious
        # noise-path problems without adding hours to the run.
        sampled = {
            ("balanced", "sparse25"),
            ("balanced", "medium50"),
            ("workload", "cubic"),
            ("workload", "er75"),
        }
        return instance_index == 0 and (suite, family) in sampled
    limit = PROFILES[profile_name]["noise_instances"]
    return instance_index < limit


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def package_versions():
    names = ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy")
    return {name: version(name) for name in names}


def mean(values):
    values = [float(v) for v in values if np.isfinite(float(v))]
    return float(statistics.fmean(values)) if values else float("nan")


def bootstrap_ci(values, reps, seed=20260924):
    values = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.size, size=(reps, values.size))
    means = values[picks].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def saved_pct(stock, hybrid):
    stock = float(stock)
    hybrid = float(hybrid)
    return 100.0 * (stock - hybrid) / stock if stock else float("nan")


def average_by_graph(rows):
    grouped = {}
    for row in rows:
        key = (row["suite"], row["family"], row["graph_id"], row["case"], row["method"])
        grouped.setdefault(key, []).append(row)

    out = {}
    numeric = (
        "native_2q",
        "cx",
        "depth",
        "twoq_depth",
        "duration_us",
        "router_seconds",
        "level3_seconds",
        "total_compile_seconds",
        "noisy_cut",
    )
    for key, items in grouped.items():
        averaged = {field: mean([item[field] for item in items]) for field in numeric}
        averaged["noise_simulated"] = any(item["noise_simulated"] for item in items)
        out[key] = averaged
    return out


def paired_rows(rows):
    averaged = average_by_graph(rows)
    pairs = []
    bases = {}
    hybrids = {}
    for key, values in averaged.items():
        suite, family, graph_id, case, method = key
        target = bases if method == "stock_level3" else hybrids
        target[(suite, family, graph_id, case)] = values

    for key in sorted(set(bases) & set(hybrids)):
        stock = bases[key]
        hybrid = hybrids[key]
        suite, family, graph_id, case = key
        noisy_delta = float("nan")
        if np.isfinite(stock["noisy_cut"]) and np.isfinite(hybrid["noisy_cut"]):
            noisy_delta = hybrid["noisy_cut"] - stock["noisy_cut"]
        pairs.append({
            "suite": suite,
            "family": family,
            "graph_id": graph_id,
            "case": case,
            "native_2q_saving_pct": saved_pct(stock["native_2q"], hybrid["native_2q"]),
            "duration_saving_pct": saved_pct(stock["duration_us"], hybrid["duration_us"]),
            "twoq_depth_saving_pct": saved_pct(stock["twoq_depth"], hybrid["twoq_depth"]),
            "depth_saving_pct": saved_pct(stock["depth"], hybrid["depth"]),
            "compile_time_change_pct": (
                100.0 * (hybrid["total_compile_seconds"] - stock["total_compile_seconds"])
                / stock["total_compile_seconds"]
                if stock["total_compile_seconds"]
                else float("nan")
            ),
            "stock_native_2q": stock["native_2q"],
            "hybrid_native_2q": hybrid["native_2q"],
            "stock_duration_us": stock["duration_us"],
            "hybrid_duration_us": hybrid["duration_us"],
            "noisy_cut_delta": noisy_delta,
        })
    return pairs


def summarize_pairs(pairs, reps):
    rows = []

    def build(scope, suite, family, case, subset):
        native = [item["native_2q_saving_pct"] for item in subset]
        low, high = bootstrap_ci(native, reps)
        diffs = [item["stock_native_2q"] - item["hybrid_native_2q"] for item in subset]
        noisy = [item["noisy_cut_delta"] for item in subset if np.isfinite(item["noisy_cut_delta"])]
        return {
            "scope": scope,
            "suite": suite,
            "family": family,
            "case": case,
            "pairs": len(subset),
            "native_2q_saving_pct": mean(native),
            "native_2q_ci_low": low,
            "native_2q_ci_high": high,
            "duration_saving_pct": mean([item["duration_saving_pct"] for item in subset]),
            "twoq_depth_saving_pct": mean([item["twoq_depth_saving_pct"] for item in subset]),
            "quantum_depth_saving_pct": mean([item["depth_saving_pct"] for item in subset]),
            "compile_time_change_pct": mean([item["compile_time_change_pct"] for item in subset]),
            "wins": sum(value > 1e-12 for value in diffs),
            "ties": sum(abs(value) <= 1e-12 for value in diffs),
            "losses": sum(value < -1e-12 for value in diffs),
            "noisy_cut_delta": mean(noisy),
            "noisy_pairs": len(noisy),
        }

    families = sorted({(p["suite"], p["family"], p["case"]) for p in pairs})
    for suite, family, case in families:
        subset = [p for p in pairs if p["suite"] == suite and p["family"] == family and p["case"] == case]
        rows.append(build("family_case", suite, family, case, subset))

    for suite in ("balanced", "workload"):
        subset = [p for p in pairs if p["suite"] == suite]
        if subset:
            rows.append(build("suite_graph_weighted", suite, "ALL", "ALL", subset))

            cells = [
                row for row in rows
                if row["scope"] == "family_case" and row["suite"] == suite
            ]
            synthetic = []
            for index, cell in enumerate(cells):
                synthetic.append({
                    "native_2q_saving_pct": cell["native_2q_saving_pct"],
                    "duration_saving_pct": cell["duration_saving_pct"],
                    "twoq_depth_saving_pct": cell["twoq_depth_saving_pct"],
                    "depth_saving_pct": cell["quantum_depth_saving_pct"],
                    "compile_time_change_pct": cell["compile_time_change_pct"],
                    "stock_native_2q": 1.0,
                    "hybrid_native_2q": 1.0 - cell["native_2q_saving_pct"] / 100.0,
                    "noisy_cut_delta": cell["noisy_cut_delta"],
                })
            rows.append(build("suite_family_balanced", suite, "ALL", "ALL", synthetic))
    return rows


def fmt(value, digits=2):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def report_text(profile_name, metadata, rows, summary):
    lines = [
        "# Hybrid Level 3 benchmark",
        "",
        "I wanted one test that makes it hard for me to accidentally cherry-pick a nice-looking result,",
        "so this run always executes both suites in one command.",
        "",
        "- **Balanced** deliberately covers very sparse through almost-complete graphs plus ring and all-to-all.",
        "- **Workload** uses regular, grid, community and Erdos-Renyi interaction graphs. It is structure-diverse,",
        "  but I am **not** pretending these are measured production frequencies from IBM hardware.",
        "",
        "The comparison is stock Qiskit `optimization_level=3` vs my local Hybrid Level 3. Both get the exact",
        "same logical PauliEvolutionGate QAOA circuit and compiler seed.",
        "",
        "## Headline",
        "",
    ]

    for suite in ("balanced", "workload"):
        family_balanced = next(
            (r for r in summary if r["scope"] == "suite_family_balanced" and r["suite"] == suite),
            None,
        )
        graph_weighted = next(
            (r for r in summary if r["scope"] == "suite_graph_weighted" and r["suite"] == suite),
            None,
        )
        if not family_balanced or not graph_weighted:
            continue
        lines.extend([
            f"### {suite.title()} suite",
            "",
            f"- Family-balanced native 2Q saving: **{fmt(family_balanced['native_2q_saving_pct'])}%**.",
            f"- Graph-weighted native 2Q saving: **{fmt(graph_weighted['native_2q_saving_pct'])}%** "
            f"(95% bootstrap CI {fmt(graph_weighted['native_2q_ci_low'])}% to {fmt(graph_weighted['native_2q_ci_high'])}%).",
            f"- Estimated hardware duration saving: **{fmt(family_balanced['duration_saving_pct'])}%**.",
            f"- 2Q-depth saving: **{fmt(family_balanced['twoq_depth_saving_pct'])}%**.",
            f"- Classical compile-time change: **{fmt(family_balanced['compile_time_change_pct'])}%** "
            "(positive means Hybrid took longer to compile).",
            f"- Native-2Q W/T/L on graph-cases: **{graph_weighted['wins']}/{graph_weighted['ties']}/{graph_weighted['losses']}**.",
            f"- Noisy MaxCut delta: **{fmt(graph_weighted['noisy_cut_delta'], 3)}** across {graph_weighted['noisy_pairs']} paired noisy graph-cases.",
            "",
        ])

    lines.extend([
        "## Per family / QAOA depth",
        "",
        "| suite | family | p | graphs | native 2Q saving | duration saving | 2Q-depth saving | W/T/L | noisy cut delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary:
        if row["scope"] != "family_case":
            continue
        lines.append(
            f"| {row['suite']} | {row['family']} | {str(row['case'])[-1]} | {row['pairs']} | "
            f"{fmt(row['native_2q_saving_pct'])}% | {fmt(row['duration_saving_pct'])}% | "
            f"{fmt(row['twoq_depth_saving_pct'])}% | {row['wins']}/{row['ties']}/{row['losses']} | "
            f"{fmt(row['noisy_cut_delta'], 3)} |"
        )

    hybrid_rows = [row for row in rows if row["method"] == "hybrid_level3"]
    modes = {}
    for row in hybrid_rows:
        modes[row["route_mode"]] = modes.get(row["route_mode"], 0) + 1

    lines.extend([
        "",
        "## What the selector actually did",
        "",
    ])
    for mode, count in sorted(modes.items()):
        lines.append(f"- `{mode}`: {count} compiler-seed runs")

    lines.extend([
        "",
        "## Correctness and realism notes",
        "",
        f"- Every compiled circuit had to pass a statevector check with amplitude error <= {STATE_TOL:g}.",
        "- Estimated duration comes from the historical FakeGuadalupeV2 target.",
        "- Noise rows use AerSimulator.from_backend(FakeGuadalupeV2), so gate/readout/relaxation-style errors from",
        "  that snapshot are included where Aer supports them.",
        "- This still does not include every real effect such as all crosstalk, calibration drift, queue behavior,",
        "  or a current device calibration. It is a hardware-calibrated local test, not a physical-QPU result.",
        f"- Profile: `{profile_name}`. Graph start: {metadata['graph_start']}. Compiler seeds: {metadata['compiler_seeds']}.",
        "",
        "## Files",
        "",
        "`raw.csv` has every compiler-seed run. `summary.csv` has the paired aggregate table. `metadata.json`",
        "contains the exact profile and package versions.",
    ])
    return "\n".join(lines) + "\n"


def run(profile_name: str, graph_start: int, output: Path | None):
    profile = PROFILES[profile_name]
    backend = FakeGuadalupeV2()
    if backend.num_qubits != N:
        raise RuntimeError("This benchmark is locked to the 16-qubit FakeGuadalupeV2 snapshot.")

    simulator = AerSimulator.from_backend(
        backend,
        method="statevector",
        max_parallel_threads=1,
        max_parallel_shots=4,
        max_memory_mb=2048,
    )

    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path(__file__).resolve().parent / "results" / f"{profile_name}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    metadata = {
        "status": "running",
        "started_utc": utc_now(),
        "profile": profile_name,
        "graph_start": graph_start,
        "qubits": N,
        "backend": "FakeGuadalupeV2 historical calibration",
        "backend_calibration_date": backend.properties().last_update_date.isoformat(),
        "compiler_seeds": list(profile["compiler_seeds"]),
        "cases": [f"qaoa_p{p}" for p in profile["cases"]],
        "balanced_instances_per_random_family": profile["balanced_instances"],
        "workload_instances_per_random_family": profile["workload_instances"],
        "shots": profile["shots"],
        "noise_policy": {
            "quick": "p1 only on the first balanced/medium50 and workload/cubic graphs",
            "standard": (
                "first balanced/sparse25, balanced/medium50, workload/cubic and "
                "workload/er75 graphs, for both p1 and p2"
            ),
            "stress": f"first {profile['noise_instances']} instance(s) of every family, for p1 and p2",
        }[profile_name],
        "beam_seconds": profile["beam_seconds"],
        "packages": package_versions(),
        "fairness": [
            "Stock and Hybrid receive the exact same logical PauliEvolutionGate QAOA circuit.",
            "Stock and Hybrid receive the same Qiskit compiler seed.",
            "Hybrid automatically falls back to stock Qiskit outside its tested QAOA/density window.",
            "The balanced and workload suites are both started by the same command.",
            "The workload suite is structure-diverse; it is not claimed to match measured production frequencies.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    specs = graph_specs(profile_name, graph_start)
    # I save raw.csv after every compiled circuit. If Windows or Python dies halfway
    # through a long run, I still have the work that already finished.
    raw_rows = []
    total_graphs = len(specs)

    try:
        for graph_number, spec in enumerate(specs, start=1):
            instance_index = spec.graph_id - graph_start
            scores = cut_scores(spec.edges)
            print(
                f"[{graph_number}/{total_graphs}] {spec.suite}/{spec.family} "
                f"density={spec.density:.3f}",
                flush=True,
            )

            for layers in profile["cases"]:
                case = f"qaoa_p{layers}"
                angle_seed = spec.graph_seed + 211 + layers
                logical, angles = qaoa_circuit(spec.edges, layers, angle_seed)
                reference_state = reference_statevector(spec.edges, angles)
                reference_probs = np.abs(reference_state) ** 2
                ideal_cut = float(np.dot(reference_probs, scores))
                do_noise = should_simulate_noise(
                    profile_name, spec.suite, spec.family, instance_index, layers
                )

                for compiler_seed in profile["compiler_seeds"]:
                    simulator_seed = spec.graph_seed + layers * 100 + compiler_seed * 1000
                    for method in ("stock_level3", "hybrid_level3"):
                        start = time.perf_counter()
                        if method == "stock_level3":
                            result = compile_stock_level3(
                                logical.copy(), backend, seed_transpiler=compiler_seed
                            )
                        else:
                            result = compile_hybrid_level3(
                                logical.copy(),
                                backend,
                                seed_transpiler=compiler_seed,
                                beam_seconds=profile["beam_seconds"],
                            )
                        wall_seconds = time.perf_counter() - start

                        compiled_state = logical_amplitudes(result.circuit, result.final_positions)
                        state_error = amplitude_error(reference_state, compiled_state)
                        compiled_probs = np.abs(compiled_state) ** 2
                        total_variation = 0.5 * float(np.abs(compiled_probs - reference_probs).sum())
                        compiled_cut = float(np.dot(compiled_probs, scores))
                        if state_error > STATE_TOL:
                            raise AssertionError(
                                f"State mismatch: {spec.suite}/{spec.family} {case} {method} "
                                f"seed={compiler_seed}, error={state_error:.3g}"
                            )
                        if total_variation > TV_TOL or abs(compiled_cut - ideal_cut) > CUT_TOL:
                            raise AssertionError(
                                f"Output drift: {spec.suite}/{spec.family} {case} {method} "
                                f"TV={total_variation:.3g}, cut={compiled_cut - ideal_cut:.3g}"
                            )

                        noisy_cut = float("nan")
                        simulator_seconds = 0.0
                        actual_shots = 0
                        if do_noise:
                            sim_start = time.perf_counter()
                            sim_result = simulator.run(
                                measured_circuit(result.circuit, result.final_positions),
                                shots=profile["shots"],
                                seed_simulator=simulator_seed,
                            ).result()
                            if not sim_result.success:
                                raise RuntimeError(f"Aer failed: {sim_result.status}")
                            simulator_seconds = time.perf_counter() - sim_start
                            noisy_cut = shot_mean(dict(sim_result.get_counts()), scores)
                            actual_shots = profile["shots"]

                        metrics = circuit_metrics(result.circuit, backend.target)
                        raw_rows.append({
                            "suite": spec.suite,
                            "family": spec.family,
                            "graph_id": spec.graph_id,
                            "graph_seed": spec.graph_seed,
                            "density": spec.density,
                            "case": case,
                            "layers": layers,
                            "method": method,
                            "compiler_seed": compiler_seed,
                            "route_mode": result.route_mode,
                            "selector_reason": result.selector_reason,
                            **metrics,
                            "router_seconds": result.router_seconds,
                            "level3_seconds": result.level3_seconds,
                            "total_compile_seconds": wall_seconds,
                            "swaps_before_level3": result.swaps_before_level3,
                            "state_error": state_error,
                            "total_variation": total_variation,
                            "ideal_cut": ideal_cut,
                            "compiled_ideal_cut": compiled_cut,
                            "noise_simulated": bool(do_noise),
                            "shots": actual_shots,
                            "noisy_cut": noisy_cut,
                            "noisy_minus_ideal": noisy_cut - ideal_cut if do_noise else float("nan"),
                            "simulator_seconds": simulator_seconds,
                            "final_positions": json.dumps(result.final_positions),
                        })
                        write_csv(output / "raw.csv", RAW_FIELDS, raw_rows)

        pairs = paired_rows(raw_rows)
        summary = summarize_pairs(pairs, profile["bootstrap_reps"])
        write_csv(output / "summary.csv", SUMMARY_FIELDS, summary)

        metadata["status"] = "complete"
        metadata["finished_utc"] = utc_now()
        metadata["raw_rows"] = len(raw_rows)
        metadata["paired_graph_cases"] = len(pairs)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (output / "REPORT.md").write_text(
            report_text(profile_name, metadata, raw_rows, summary), encoding="utf-8"
        )
        return output
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["failed_utc"] = utc_now()
        metadata["error"] = repr(exc)
        metadata["raw_rows"] = len(raw_rows)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if raw_rows:
            write_csv(output / "raw.csv", RAW_FIELDS, raw_rows)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    parser.add_argument("--graph-start", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.graph_start is None:
        # Separate ranges keep quick results from leaking into the bigger tests I care about.
        default_starts = {"quick": 70000, "standard": 80000, "stress": 90000}
        args.graph_start = default_starts[args.profile]
    if args.graph_start < 0:
        parser.error("--graph-start must be non-negative")

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
    warnings.filterwarnings("ignore", message="splu converted its input to CSC format")
    warnings.filterwarnings(
        "ignore", message="spsolve is more efficient when sparse b is in the CSC matrix format"
    )

    output = run(args.profile, args.graph_start, args.output)
    print()
    print(f"Finished. Results: {output.resolve()}")
    print(f"Main report: {(output / 'REPORT.md').resolve()}")


if __name__ == "__main__":
    main()
