"""Synthetic circuit generators for benchmarking.

This module provides functions to generate the 7 circuit types used in the
thesis scaling study (Section 4.3, Figure 4.1).

References:
    - Qiskit random circuits: https://docs.quantum.ibm.com/api/qiskit/circuit_random
    - Pozzi 2020 (full_layer): Using Reinforcement Learning for Qubit Routing
    - Maslov 2008, Saeedi 2011 (circuit patterns)
"""

import enum
import random

import qiskit.circuit
import qiskit.circuit.random


class CircuitType(enum.StrEnum):
    """Circuit types for synthetic benchmarking.

    Attributes:
        RANDOM_CLIFFORD: Random Clifford gates (CX + H).
        LINEAR_CHAIN: Sequential CX chain pattern.
        RING: Complete cycle CX pattern.
        STAR: Hub-and-spoke CX pattern.
        QFT_LIKE: QFT-inspired all-to-all pattern.
        FULL_LAYER: Dense parallel CX layers (Pozzi 2020).
        DEPTH_CONTROLLED: Random with controlled circuit depth.
    """

    RANDOM_CLIFFORD = "random_clifford"
    LINEAR_CHAIN = "linear_chain"
    RING = "ring"
    STAR = "star"
    QFT_LIKE = "qft_like"
    FULL_LAYER = "full_layer"
    DEPTH_CONTROLLED = "depth_controlled"


# Gate name constants for Clifford circuit generation
GATE_CX = "cx"
GATE_H = "h"
CLIFFORD_GATE_SET: list[str] = [GATE_CX, GATE_H]


def generate_random_clifford(
    num_qubits: int,
    num_gates: int,
    seed: int,
) -> qiskit.circuit.QuantumCircuit:
    """Generate random Clifford circuit with CX and H gates.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_gates: Total number of gates to generate.
        seed: Random seed for reproducibility.

    Returns:
        QuantumCircuit with random Clifford gates.

    Reference:
        https://docs.quantum.ibm.com/api/qiskit/circuit_random#random_clifford_circuit
    """
    circuit = qiskit.circuit.random.random_clifford_circuit(
        num_qubits,
        num_gates,
        CLIFFORD_GATE_SET,
        seed=seed,
    )
    return circuit


def generate_linear_chain(
    num_qubits: int,
    num_layers: int,
) -> qiskit.circuit.QuantumCircuit:
    """Generate linear chain circuit with sequential CX gates.

    Creates CX gates connecting adjacent qubits in sequence: 0→1→2→...→(n-1).

    Args:
        num_qubits: Number of qubits in the circuit.
        num_layers: Number of times to repeat the chain pattern.

    Returns:
        QuantumCircuit with linear chain CX pattern.
    """
    circuit = qiskit.circuit.QuantumCircuit(num_qubits)

    for _ in range(num_layers):
        for qubit_index in range(num_qubits - 1):
            circuit.cx(qubit_index, qubit_index + 1)

    return circuit


def generate_ring(
    num_qubits: int,
    num_layers: int,
) -> qiskit.circuit.QuantumCircuit:
    """Generate ring circuit with cyclic CX pattern.

    Creates CX gates forming a complete cycle: 0→1→2→...→(n-1)→0.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_layers: Number of times to repeat the ring pattern.

    Returns:
        QuantumCircuit with cyclic CX pattern.
    """
    circuit = qiskit.circuit.QuantumCircuit(num_qubits)

    for _ in range(num_layers):
        for qubit_index in range(num_qubits):
            target_qubit = (qubit_index + 1) % num_qubits
            circuit.cx(qubit_index, target_qubit)

    return circuit


def generate_star(
    num_qubits: int,
    num_layers: int,
) -> qiskit.circuit.QuantumCircuit:
    """Generate star circuit with hub-and-spoke CX pattern.

    Qubit 0 (hub) connects to all other qubits (spokes).

    Args:
        num_qubits: Number of qubits in the circuit.
        num_layers: Number of times to repeat the star pattern.

    Returns:
        QuantumCircuit with hub-and-spoke CX pattern.
    """
    circuit = qiskit.circuit.QuantumCircuit(num_qubits)

    hub_qubit = 0
    for _ in range(num_layers):
        for spoke_qubit in range(1, num_qubits):
            circuit.cx(hub_qubit, spoke_qubit)

    return circuit


def generate_qft_like(
    num_qubits: int,
    num_layers: int,
) -> qiskit.circuit.QuantumCircuit:
    """Generate QFT-like circuit with all-to-all connectivity.

    Each qubit interacts with all subsequent qubits, creating dense
    connectivity requirements.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_layers: Number of times to repeat the QFT-like pattern.

    Returns:
        QuantumCircuit with all-to-all CX pattern.
    """
    circuit = qiskit.circuit.QuantumCircuit(num_qubits)

    for _ in range(num_layers):
        for control_qubit in range(num_qubits):
            for target_qubit in range(control_qubit + 1, num_qubits):
                circuit.cx(control_qubit, target_qubit)

    return circuit


def generate_full_layer(
    num_qubits: int,
    num_layers: int,
) -> qiskit.circuit.QuantumCircuit:
    """Generate full-layer circuit with dense parallel CX layers.

    Each layer has maximum parallel CX gates, pairing adjacent qubits.
    Alternates between even-odd and odd-even pairings.

    Reference:
        Pozzi 2020: Using Reinforcement Learning for Qubit Routing

    Args:
        num_qubits: Number of qubits in the circuit.
        num_layers: Number of dense layers to generate.

    Returns:
        QuantumCircuit with dense parallel CX layers.
    """
    circuit = qiskit.circuit.QuantumCircuit(num_qubits)

    for layer_index in range(num_layers):
        # Alternate between even-odd (0,1), (2,3)... and odd-even (1,2), (3,4)...
        start_offset = layer_index % 2

        for qubit_index in range(start_offset, num_qubits - 1, 2):
            circuit.cx(qubit_index, qubit_index + 1)

    return circuit


def generate_depth_controlled(
    num_qubits: int,
    target_depth: int,
    seed: int,
) -> qiskit.circuit.QuantumCircuit:
    """Generate random circuit with controlled depth.

    Each depth layer randomly pairs qubits for CX gates.

    Args:
        num_qubits: Number of qubits in the circuit.
        target_depth: Target circuit depth (number of layers).
        seed: Random seed for reproducibility.

    Returns:
        QuantumCircuit with random gates at controlled depth.
    """
    rng = random.Random(seed)
    circuit = qiskit.circuit.QuantumCircuit(num_qubits)

    for _ in range(target_depth):
        available_qubits = list(range(num_qubits))
        rng.shuffle(available_qubits)

        # Pair up qubits (leave one unpaired if odd number)
        num_pairs = num_qubits // 2
        for pair_index in range(num_pairs):
            control_qubit = available_qubits[2 * pair_index]
            target_qubit = available_qubits[2 * pair_index + 1]
            circuit.cx(control_qubit, target_qubit)

    return circuit


# Mapping from circuit type enum to generator functions
CIRCUIT_GENERATORS = {
    CircuitType.RANDOM_CLIFFORD: generate_random_clifford,
    CircuitType.LINEAR_CHAIN: generate_linear_chain,
    CircuitType.RING: generate_ring,
    CircuitType.STAR: generate_star,
    CircuitType.QFT_LIKE: generate_qft_like,
    CircuitType.FULL_LAYER: generate_full_layer,
    CircuitType.DEPTH_CONTROLLED: generate_depth_controlled,
}
