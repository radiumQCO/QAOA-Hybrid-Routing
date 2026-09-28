"""A quick local check of the installed Qiskit routing plugin."""

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp


n = 16
backend = GenericBackendV2(n, coupling_map=[[i, i + 1] for i in range(n - 1)], seed=7)
edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
cost = SparsePauliOp.from_sparse_list(
    [("ZZ", [a, b], 1.0) for a, b in edges], num_qubits=n,
)
circuit = QuantumCircuit(n)
circuit.h(range(n))
circuit.append(PauliEvolutionGate(cost, time=0.37), range(n))
circuit.rx(0.42, range(n))
circuit.measure_all()

compiled = transpile(
    circuit, backend=backend, optimization_level=3,
    routing_method="qaoa_hybrid", seed_transpiler=11,
)
print("Router:", compiled.metadata["qaoa_hybrid_route"])
print("Native 2Q gates:", sum(inst.operation.num_qubits == 2 for inst in compiled.data))
