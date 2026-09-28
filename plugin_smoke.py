"""Small installed-package smoke check, including compiled 16q correctness."""

import argparse
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from routing_metrics import twoq_depth


def _native_twoq(circuit, target):
    count = 0
    for instruction in circuit.data:
        if instruction.operation.num_qubits != 2:
            continue
        sites = tuple(circuit.find_bit(bit).index for bit in instruction.qubits)
        assert target.instruction_supported(
            operation_name=instruction.operation.name, qargs=sites,
        ), (instruction.operation.name, sites)
        count += 1
    return count


def _ibm_regression():
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    backend = FakeBrisbane()
    for n, graph_id, expected in ((16, 991234, 309), (64, 100700, 5629)):
        seed = graph_id if n == 16 else graph_id + 1_000_000 * n
        possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
        picked = np.random.default_rng(seed).choice(
            len(possible), round(len(possible) * 0.5), replace=False,
        )
        edges = sorted(possible[int(index)] for index in picked)
        gamma, beta = np.random.default_rng(567 if n == 16 else seed + 212).uniform(
            0.15, 0.95, size=(1, 2),
        )[0]
        operator = SparsePauliOp.from_sparse_list(
            [("ZZ", [a, b], 1.0) for a, b in edges], num_qubits=n,
        )
        circuit = QuantumCircuit(n)
        circuit.h(range(n))
        circuit.append(PauliEvolutionGate(operator, time=float(gamma)), range(n))
        circuit.rx(2 * float(beta), range(n))
        compiled = transpile(
            circuit, backend=backend, optimization_level=3,
            routing_method="qaoa_hybrid", seed_transpiler=11,
        )
        count = _native_twoq(compiled, backend.target)
        assert compiled.metadata["qaoa_hybrid_route"] == "hybrid"
        assert count == expected, (n, count, expected)
        if n == 64:
            assert twoq_depth(compiled) == 185
        print(f"IBM snapshot {n}q: {count} native 2Q, depth {twoq_depth(compiled)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ibm", action="store_true", help="also check saved IBM snapshot cases")
    args = parser.parse_args()
    assert "qaoa_hybrid" in list_stage_plugins("routing")
    n = 16
    backend = GenericBackendV2(
        n, coupling_map=[[i, i + 1] for i in range(n - 1)], seed=7,
    )
    operator = SparsePauliOp.from_sparse_list(
        [("ZZ", [a, b], 1.0) for a in range(n) for b in range(a + 1, n)],
        num_qubits=n,
    )
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    circuit.append(PauliEvolutionGate(operator, time=0.37), range(n))
    circuit.rx(0.42, range(n))
    circuit.measure_all()
    compiled = transpile(
        circuit, backend=backend, optimization_level=3,
        routing_method="qaoa_hybrid", seed_transpiler=11,
    )
    assert compiled.metadata["qaoa_hybrid_route"] == "hybrid"
    assert compiled.count_ops().get("measure", 0) == n
    twoq = _native_twoq(compiled, backend.target)
    final_sites = compiled.layout.final_index_layout()[:n]
    assert len(set(final_sites)) == n
    for instruction in compiled.data:
        if instruction.operation.name == "measure":
            site = compiled.find_bit(instruction.qubits[0]).index
            classical = compiled.find_bit(instruction.clbits[0]).index
            assert site == final_sites[classical]

    # Remove idle device wires before a 16q statevector comparison.
    used = sorted({
        compiled.find_bit(bit).index
        for instruction in compiled.data if instruction.operation.name != "measure"
        for bit in instruction.qubits
    })
    assert len(used) == n
    local = {site: index for index, site in enumerate(used)}
    reduced = QuantumCircuit(n, global_phase=compiled.global_phase)
    for instruction in compiled.data:
        if instruction.operation.name != "measure":
            reduced.append(instruction.operation, [local[compiled.find_bit(bit).index]
                                                   for bit in instruction.qubits])
    ideal_circuit = QuantumCircuit(n)
    ideal_circuit.h(range(n))
    for a in range(n):
        for b in range(a + 1, n):
            ideal_circuit.rzz(0.74, a, b)
    ideal_circuit.rx(0.42, range(n))
    ideal = Statevector.from_instruction(ideal_circuit).data
    observed = Statevector.from_instruction(reduced).data
    mapped = np.empty_like(observed)
    for logical_index in range(1 << n):
        physical_index = sum(
            ((logical_index >> bit) & 1) << local[final_sites[bit]]
            for bit in range(n)
        )
        mapped[logical_index] = observed[physical_index]
    phase = np.vdot(ideal, mapped)
    assert abs(phase) > 0.999999
    error = float(np.max(np.abs(mapped / (phase / abs(phase)) - ideal)))
    assert error < 1e-8, error

    generic = QuantumCircuit(3, 3)
    generic.h(0)
    generic.cx(0, 2)
    generic.measure(range(3), range(3))
    other = transpile(
        generic, backend=backend, optimization_level=3,
        routing_method="qaoa_hybrid", seed_transpiler=11,
    )
    assert other.metadata["qaoa_hybrid_route"] == "sabre_fallback"
    _native_twoq(other, backend.target)

    no_durations = GenericBackendV2(
        n, coupling_map=[[i, i + 1] for i in range(n - 1)],
        basis_gates=["cx", "id", "rz", "sx", "x", "measure", "reset"],
        noise_info=False,
    )
    unsupported = transpile(
        circuit, backend=no_durations, optimization_level=3,
        routing_method="qaoa_hybrid", seed_transpiler=11,
    )
    assert unsupported.metadata["qaoa_hybrid_route"] == "sabre_fallback"
    print(f"PASS: plugin discovery, {twoq} native 2Q, final mapping, measures, "
          f"compiled state error {error:.2e}, safe fallbacks")
    if args.ibm:
        _ibm_regression()


if __name__ == "__main__":
    main()
