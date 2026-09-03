import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.qiskit_route_construction import (
    add_route_register_superposition,
    add_validity_controlled_cost_phase_oracle,
    add_validity_oracle_compute,
    add_validity_oracle_uncompute,
    build_qiskit_tsp_construction_circuit,
    build_route_layout,
    classical_route_cost,
    classical_route_validity,
    route_basis_index,
    wrap_to_pi,
)


def _test_cost_matrix() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.2, 0.5, 0.4],
            [0.3, 0.0, 0.1, 0.6],
            [0.7, 0.4, 0.0, 0.2],
            [0.8, 0.3, 0.9, 0.0],
        ]
    )


def _parity_bits(route, n):
    return [sum(1 for city in route if city == i) % 2 for i in range(n - 1)]


def _amp_phase(statevector: np.ndarray, idx: int) -> float:
    return float(np.angle(statevector[idx]))


def test_validity_oracle_marks_good_for_permutations_only():
    n = 4
    layout = build_route_layout(n=n, start_node=n - 1)
    circuit = QuantumCircuit(layout.num_qubits)
    add_route_register_superposition(circuit, layout)
    add_validity_oracle_compute(circuit, layout)

    state = np.asarray(Statevector.from_instruction(circuit).data)
    expected_abs = 1.0 / math.sqrt(2 ** len(layout.route_wires))

    for route in itertools.product(range(n - 1), repeat=layout.steps):
        idx = route_basis_index(route, layout)
        parity = _parity_bits(route, n)
        for wire, bit in zip(layout.state_wires, parity):
            if bit:
                idx |= 1 << wire
        is_valid = classical_route_validity(route, n=n)
        if is_valid:
            idx |= 1 << layout.good_wire
        amp = state[idx]
        assert np.isclose(abs(amp), expected_abs, atol=1e-12)


def test_validity_controlled_phase_changes_only_valid_routes():
    cost_matrix = _test_cost_matrix()
    n = cost_matrix.shape[0]
    circuit, layout = build_qiskit_tsp_construction_circuit(cost_matrix=cost_matrix, start_node=n - 1)
    state = np.asarray(Statevector.from_instruction(circuit).data)
    ancillas = layout.state_wires + [layout.good_wire, layout.phase_wire]
    leaked_probability = 0.0
    for idx, amp in enumerate(state):
        if abs(amp) < 1e-14:
            continue
        if any(((idx >> wire) & 1) for wire in ancillas):
            leaked_probability += float(abs(amp) ** 2)
    assert np.isclose(leaked_probability, 0.0, atol=1e-12)

    invalid_routes = [r for r in itertools.product(range(n - 1), repeat=layout.steps) if not classical_route_validity(r, n=n)]
    invalid_phases = [_amp_phase(state, route_basis_index(route, layout)) for route in invalid_routes]
    for phase in invalid_phases[1:]:
        assert np.isclose(wrap_to_pi(phase - invalid_phases[0]), 0.0, atol=1e-10)

    valid_routes = [r for r in itertools.product(range(n - 1), repeat=layout.steps) if classical_route_validity(r, n=n)]
    ref_route = valid_routes[0]
    ref_phase = _amp_phase(state, route_basis_index(ref_route, layout))
    ref_cost = classical_route_cost(cost_matrix, ref_route, start_node=layout.start_node)
    for route in valid_routes[1:]:
        phase = _amp_phase(state, route_basis_index(route, layout))
        cost = classical_route_cost(cost_matrix, route, start_node=layout.start_node)
        assert np.isclose(
            wrap_to_pi((phase - ref_phase) - (cost - ref_cost)),
            0.0,
            atol=1e-10,
        )


def test_uncomputation_resets_all_ancilla_registers():
    cost_matrix = _test_cost_matrix()
    n = cost_matrix.shape[0]
    layout = build_route_layout(n=n, start_node=n - 1)
    circuit = QuantumCircuit(layout.num_qubits)
    add_route_register_superposition(circuit, layout)
    add_validity_oracle_compute(circuit, layout)
    add_validity_controlled_cost_phase_oracle(circuit, layout, cost_matrix)
    add_validity_oracle_uncompute(circuit, layout)

    state = np.asarray(Statevector.from_instruction(circuit).data)
    ancillas = layout.state_wires + [layout.good_wire, layout.phase_wire]
    leaked_probability = 0.0
    for idx, amp in enumerate(state):
        if abs(amp) < 1e-14:
            continue
        if any(((idx >> wire) & 1) for wire in ancillas):
            leaked_probability += float(abs(amp) ** 2)
    assert np.isclose(leaked_probability, 0.0, atol=1e-12)
