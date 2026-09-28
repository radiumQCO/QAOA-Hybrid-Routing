"""Small routing metrics shared by the installed plugin and benchmarks."""


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
