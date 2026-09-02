import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit


@dataclass(frozen=True)
class RouteLayout:
    n: int
    start_node: int
    steps: int
    bits_per_step: int
    route_city_nodes: list[int]
    route_wires_by_step: list[list[int]]
    route_wires: list[int]
    state_wires: list[int]
    good_wire: int
    phase_wire: int
    num_qubits: int


def build_route_layout(n: int, start_node: int | None = None) -> RouteLayout:
    if n < 3:
        raise ValueError("n must be at least 3.")
    if start_node is None:
        start_node = n - 1
    if not 0 <= start_node < n:
        raise ValueError("start_node must be in [0, n-1].")

    steps = n - 1
    bits_per_step = int(math.ceil(math.log2(n - 1)))
    route_city_nodes = [node for node in range(n) if node != start_node]
    route_wires_by_step: list[list[int]] = []
    cursor = 0
    for _ in range(steps):
        step_wires = list(range(cursor, cursor + bits_per_step))
        route_wires_by_step.append(step_wires)
        cursor += bits_per_step

    route_wires = [w for step in route_wires_by_step for w in step]
    state_wires = list(range(cursor, cursor + (n - 1)))
    cursor += n - 1
    good_wire = cursor
    cursor += 1
    phase_wire = cursor
    cursor += 1

    return RouteLayout(
        n=n,
        start_node=start_node,
        steps=steps,
        bits_per_step=bits_per_step,
        route_city_nodes=route_city_nodes,
        route_wires_by_step=route_wires_by_step,
        route_wires=route_wires,
        state_wires=state_wires,
        good_wire=good_wire,
        phase_wire=phase_wire,
        num_qubits=cursor,
    )


def _mcx_on_value(circuit: QuantumCircuit, controls: list[int], bitstring: str, target: int) -> None:
    for wire, bit in zip(controls, bitstring):
        if bit == "0":
            circuit.x(wire)
    circuit.mcx(controls, target)
    for wire, bit in zip(controls, bitstring):
        if bit == "0":
            circuit.x(wire)


def add_route_register_superposition(circuit: QuantumCircuit, layout: RouteLayout) -> None:
    for wire in layout.route_wires:
        circuit.h(wire)


def add_validity_oracle_compute(circuit: QuantumCircuit, layout: RouteLayout) -> None:
    for step_wires in layout.route_wires_by_step:
        for city in range(layout.n - 1):
            bits = format(city, f"0{layout.bits_per_step}b")
            _mcx_on_value(circuit, step_wires, bits, layout.state_wires[city])
    circuit.mcx(layout.state_wires, layout.good_wire)


def add_validity_oracle_uncompute(circuit: QuantumCircuit, layout: RouteLayout) -> None:
    circuit.mcx(layout.state_wires, layout.good_wire)
    for step_wires in reversed(layout.route_wires_by_step):
        for city in reversed(range(layout.n - 1)):
            bits = format(city, f"0{layout.bits_per_step}b")
            _mcx_on_value(circuit, step_wires, bits, layout.state_wires[city])


def _apply_phase_kickback(
    circuit: QuantumCircuit,
    controls: list[int],
    control_bits: str,
    phase_wire: int,
    angle: float,
) -> None:
    for wire, bit in zip(controls, control_bits):
        if bit == "0":
            circuit.x(wire)
    circuit.mcx(controls, phase_wire)
    circuit.p(float(angle), phase_wire)
    circuit.mcx(controls, phase_wire)
    for wire, bit in zip(controls, control_bits):
        if bit == "0":
            circuit.x(wire)


def add_validity_controlled_cost_phase_oracle(
    circuit: QuantumCircuit,
    layout: RouteLayout,
    cost_matrix: np.ndarray,
) -> None:
    if cost_matrix.shape != (layout.n, layout.n):
        raise ValueError("cost_matrix shape must be (n, n).")

    first_step = layout.route_wires_by_step[0]
    for city in range(layout.n - 1):
        actual_city = layout.route_city_nodes[city]
        city_bits = format(city, f"0{layout.bits_per_step}b")
        controls = [layout.good_wire] + first_step
        bits = "1" + city_bits
        _apply_phase_kickback(
            circuit=circuit,
            controls=controls,
            control_bits=bits,
            phase_wire=layout.phase_wire,
            angle=float(cost_matrix[layout.start_node, actual_city]),
        )

    for t in range(layout.steps - 1):
        from_step = layout.route_wires_by_step[t]
        to_step = layout.route_wires_by_step[t + 1]
        controls = [layout.good_wire] + from_step + to_step
        for i in range(layout.n - 1):
            actual_i = layout.route_city_nodes[i]
            i_bits = format(i, f"0{layout.bits_per_step}b")
            for j in range(layout.n - 1):
                actual_j = layout.route_city_nodes[j]
                j_bits = format(j, f"0{layout.bits_per_step}b")
                _apply_phase_kickback(
                    circuit=circuit,
                    controls=controls,
                    control_bits="1" + i_bits + j_bits,
                    phase_wire=layout.phase_wire,
                    angle=float(cost_matrix[actual_i, actual_j]),
                )

    last_step = layout.route_wires_by_step[-1]
    for city in range(layout.n - 1):
        actual_city = layout.route_city_nodes[city]
        city_bits = format(city, f"0{layout.bits_per_step}b")
        controls = [layout.good_wire] + last_step
        bits = "1" + city_bits
        _apply_phase_kickback(
            circuit=circuit,
            controls=controls,
            control_bits=bits,
            phase_wire=layout.phase_wire,
            angle=float(cost_matrix[actual_city, layout.start_node]),
        )


def build_qiskit_tsp_construction_circuit(
    cost_matrix: np.ndarray,
    start_node: int | None = None,
) -> tuple[QuantumCircuit, RouteLayout]:
    n = int(cost_matrix.shape[0])
    layout = build_route_layout(n=n, start_node=start_node)
    circuit = QuantumCircuit(layout.num_qubits)

    add_route_register_superposition(circuit, layout)
    add_validity_oracle_compute(circuit, layout)
    add_validity_controlled_cost_phase_oracle(circuit, layout, cost_matrix)
    add_validity_oracle_uncompute(circuit, layout)

    return circuit, layout


def classical_route_validity(route: tuple[int, ...] | list[int], n: int) -> bool:
    return sorted(route) == list(range(n - 1))


def classical_route_cost(cost_matrix: np.ndarray, route: tuple[int, ...] | list[int], start_node: int) -> float:
    n = int(cost_matrix.shape[0])
    route_city_nodes = [node for node in range(n) if node != start_node]
    route_nodes = [route_city_nodes[int(city)] for city in route]
    cost = float(cost_matrix[start_node, route_nodes[0]])
    for i in range(len(route_nodes) - 1):
        cost += float(cost_matrix[route_nodes[i], route_nodes[i + 1]])
    cost += float(cost_matrix[route_nodes[-1], start_node])
    return float(cost)


def route_basis_index(route: tuple[int, ...] | list[int], layout: RouteLayout) -> int:
    idx = 0
    for step_wires, city in zip(layout.route_wires_by_step, route):
        bits = format(int(city), f"0{layout.bits_per_step}b")
        for wire, bit in zip(step_wires, bits):
            if bit == "1":
                idx |= 1 << wire
    return idx


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def build_route_phase_table(
    statevector: np.ndarray,
    layout: RouteLayout,
    cost_matrix: np.ndarray,
) -> pd.DataFrame:
    """Build a logical-route table with tour labels, validity, phase, expected phase, and probability."""
    rows = []
    for route in itertools.product(range(layout.n - 1), repeat=layout.steps):
        idx = route_basis_index(route, layout)
        amp = statevector[idx]
        validity = int(classical_route_validity(route, n=layout.n))
        expected_phi = (
            round(classical_route_cost(cost_matrix, route, layout.start_node), 8)
            if validity
            else np.nan
        )
        rows.append(
            {
                "tour": list(route),
                "validity": validity,
                "phi": round(wrap_to_pi(float(np.angle(amp))), 8),
                "expected_phi": expected_phi,
                "prob": round(float(np.abs(amp) ** 2), 8),
            }
        )
    return pd.DataFrame(rows)
