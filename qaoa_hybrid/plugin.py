"""Qiskit routing-stage adapter for QAOA Hybrid."""

from __future__ import annotations

import math
from types import SimpleNamespace

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import Layout, PassManager, TransformationPass
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin


def _prepared_qaoa(dag, properties):
    """Recover the supported QAOA layer structure after Qiskit's init/layout stages."""
    layout = properties["layout"]
    original_indices = properties["original_qubit_indices"]
    if layout is None or original_indices is None or properties["final_layout"] is not None:
        return None
    if properties["virtual_permutation_layout"] is not None:
        return None
    active_sites = {
        dag.find_bit(bit).index
        for node in dag.topological_op_nodes() if node.op.name != "barrier"
        for bit in node.qargs
    }
    original_bits = sorted(
        (layout[site] for site in active_sites), key=original_indices.__getitem__,
    )
    n = len(original_bits)
    if not 16 <= n <= 64 or len(dag.qubits) < n:
        return None
    physical_to_logical = {layout[bit]: index for index, bit in enumerate(original_bits)}
    if len(physical_to_logical) != n:
        return None

    hadamards = [False] * n
    mixer_angles = [[] for _ in range(n)]
    layers = []
    measurements = []
    measured = set()
    for node in dag.topological_op_nodes():
        name = node.op.name
        if name == "barrier":
            continue
        sites = [dag.find_bit(bit).index for bit in node.qargs]
        if any(site not in physical_to_logical for site in sites):
            return None
        logical = [physical_to_logical[site] for site in sites]
        if any(bit in measured for bit in logical):
            return None
        if name == "h" and len(logical) == 1 and not node.cargs:
            bit = logical[0]
            if hadamards[bit] or mixer_angles[bit]:
                return None
            hadamards[bit] = True
        elif name == "rzz" and len(logical) == 2 and not node.cargs:
            a, b = sorted(logical)
            layer = len(mixer_angles[a])
            if not hadamards[a] or not hadamards[b] or layer != len(mixer_angles[b]):
                return None
            try:
                angle = float(node.op.params[0])
            except (TypeError, ValueError):
                return None
            if not math.isfinite(angle):
                return None
            while len(layers) <= layer:
                layers.append({})
            if (a, b) in layers[layer]:
                return None
            layers[layer][a, b] = angle
        elif name == "rx" and len(logical) == 1 and not node.cargs:
            bit = logical[0]
            layer = len(mixer_angles[bit])
            if not hadamards[bit] or layer >= len(layers):
                return None
            try:
                angle = float(node.op.params[0])
            except (TypeError, ValueError):
                return None
            if not math.isfinite(angle):
                return None
            mixer_angles[bit].append(angle)
        elif name == "measure" and len(logical) == 1 and len(node.cargs) == 1:
            measurements.append((logical[0], node.op, node.cargs[0]))
            measured.add(logical[0])
        else:
            return None

    if not all(hadamards) or not layers:
        return None
    depth = len(layers)
    if any(len(angles) != depth for angles in mixer_angles):
        return None
    first_pairs = set(layers[0])
    if not first_pairs or any(set(layer) != first_pairs for layer in layers):
        return None
    density = len(first_pairs) / (n * (n - 1) / 2)
    if density < 0.45:
        return None

    compact = QuantumCircuit(n, global_phase=dag.global_phase)
    compact.h(range(n))
    for layer_index, pairs in enumerate(layers):
        operator = SparsePauliOp.from_sparse_list(
            [("ZZ", list(pair), angle / 2) for pair, angle in sorted(pairs.items())],
            num_qubits=n,
        )
        compact.append(PauliEvolutionGate(operator, time=1), range(n))
        for bit in range(n):
            compact.rx(mixer_angles[bit][layer_index], bit)
    return compact, original_bits, measurements


def _has_gate_durations(target):
    if target is None:
        return False
    for name in target.operation_names:
        operation = target.operation_from_name(name)
        if operation.num_qubits not in (1, 2) or name in {"measure", "reset", "delay"}:
            continue
        if any(properties is None or properties.duration is None
               for properties in target[name].values()):
            return False
    return True


class _CheckHybridShape(TransformationPass):
    def __init__(self, target):
        super().__init__()
        self.enabled = _has_gate_durations(target)

    def run(self, dag):
        prepared = _prepared_qaoa(dag, self.property_set) if self.enabled else None
        self.property_set["qaoa_hybrid_prepared"] = prepared
        return dag


class _RouteHybrid(TransformationPass):
    def __init__(self, target, seed):
        super().__init__()
        self.target = target
        self.seed = 0 if seed is None else int(seed)

    def run(self, dag):
        import hybrid_level3 as v1
        from hybrid_level3_v2 import (
            LayoutSearchFailed, compare_variants, compile_hybrid_dense_v2,
        )
        from layout_seed import stock_initial_layout

        compact, original_bits, measurements = self.property_set["qaoa_hybrid_prepared"]
        backend = SimpleNamespace(
            target=self.target,
            coupling_map=self.target.build_coupling_map(),
            num_qubits=self.target.num_qubits,
        )
        n = compact.num_qubits
        density = v1.inspect_qaoa_shape(compact).density
        if n < 24 and density < 0.65:
            preferred = stock_initial_layout(compact, backend, seed_transpiler=self.seed)
            try:
                variants, candidates = compare_variants(
                    compact, backend, seed_transpiler=self.seed, layouts=1,
                    beam_seconds=1.0, preferred_layout=preferred,
                )
            except LayoutSearchFailed:
                self.property_set["qaoa_hybrid_prepared"] = None
                return dag
            chosen = next(candidate for candidate in candidates
                          if candidate.compiled is variants["tournament"])
        else:
            # The faster line search applies only above 24 logical qubits.
            if n > 24:
                from hybrid_line_next import compile_hybrid_line_next

                compiler = compile_hybrid_line_next
            else:
                compiler = compile_hybrid_dense_v2
            _, chosen = compiler(
                compact, backend, seed_transpiler=self.seed, return_candidate=True,
            )
        routed = chosen.routed
        if len(set(routed.final_positions)) != compact.num_qubits:
            raise ValueError("Hybrid returned an invalid final qubit mapping")

        output = dag.copy_empty_like()
        output.global_phase = routed.circuit.global_phase
        for instruction in routed.circuit.data:
            indices = [routed.circuit.find_bit(bit).index for bit in instruction.qubits]
            output.apply_operation_back(
                instruction.operation, [output.qubits[index] for index in indices], (),
            )
        for logical, operation, clbit in measurements:
            output.apply_operation_back(
                operation, [output.qubits[routed.final_positions[logical]]], [clbit],
            )

        # The line search chooses a new start; record that start and the SWAP permutation.
        previous = self.property_set["layout"]
        initial = {bit: chosen.layout[index] for index, bit in enumerate(original_bits)}
        free_sites = set(range(len(dag.qubits))) - set(initial.values())
        for bit, site in previous.get_virtual_bits().items():
            if bit not in initial and site in free_sites:
                initial[bit] = site
                free_sites.remove(site)
        for bit in previous.get_virtual_bits():
            if bit not in initial:
                site = min(free_sites)
                initial[bit] = site
                free_sites.remove(site)
        self.property_set["layout"] = Layout(initial)

        at = list(range(len(dag.qubits)))
        for instruction in routed.circuit.data:
            if instruction.operation.name == "swap":
                a, b = (routed.circuit.find_bit(bit).index for bit in instruction.qubits)
                at[a], at[b] = at[b], at[a]
        permutation = [0] * len(at)
        for final_site, initial_site in enumerate(at):
            permutation[initial_site] = final_site
        if any(permutation[start] != end for start, end in
               zip(chosen.layout, routed.final_positions)):
            raise ValueError("Hybrid SWAPs disagree with the final qubit mapping")
        if len(set(permutation)) != len(permutation):
            raise ValueError("Hybrid returned a non-permutation")
        self.property_set["final_layout"] = Layout(
            {bit: permutation[index] for index, bit in enumerate(output.qubits)}
        )
        original_indices = self.property_set["original_qubit_indices"]
        self.property_set["virtual_permutation_layout"] = Layout(original_indices)
        output.metadata = dict(output.metadata or {})
        output.metadata["qaoa_hybrid_route"] = "hybrid"
        return output


class _MarkFallback(TransformationPass):
    def run(self, dag):
        dag.metadata = dict(dag.metadata or {})
        dag.metadata["qaoa_hybrid_route"] = "sabre_fallback"
        return dag


class HybridRoutingPlugin(PassManagerStagePlugin):
    """Route supported QAOA layers with Hybrid; use Sabre for other circuits."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        target = pass_manager_config.target
        coupling = target if target is not None else pass_manager_config.coupling_map
        seed = pass_manager_config.seed_transpiler
        manager = PassManager()
        manager.append(_CheckHybridShape(target))
        if target is not None:
            manager.append(ConditionalController(
                _RouteHybrid(target, seed),
                condition=lambda props: props["qaoa_hybrid_prepared"] is not None,
            ))
        manager.append(ConditionalController(
            [SabreSwap(coupling, seed=seed, trials=4), _MarkFallback()],
            condition=lambda props: props["qaoa_hybrid_prepared"] is None or target is None,
        ))
        return manager
