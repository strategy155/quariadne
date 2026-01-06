"""Pytest fixtures for Quariadne router tests.

This module provides reusable fixtures for testing quantum circuit routing:
- Coupling map fixtures using Qiskit fake backends
- Circuit generation using Qiskit QuantumCircuit
- Conversion utilities using AbstractQuantumCircuit.from_qiskit_circuit()

Ref: https://docs.pytest.org/en/stable/reference/fixtures.html
"""

import networkx as nx
import pytest
import qiskit
import qiskit.circuit.random
import qiskit_ibm_runtime.fake_provider

import quariadne.circuit
import quariadne.milp


# =============================================================================
# Gate Name Constants
# =============================================================================
# Standard gate names from Qiskit's circuit library
# Ref: https://docs.quantum.ibm.com/api/qiskit/circuit#standard-gates

CONTROLLED_X_GATE_NAME = "cx"
CONTROLLED_Z_GATE_NAME = "cz"
HADAMARD_GATE_NAME = "h"
SWAP_GATE_NAME = "swap"


# =============================================================================
# Test Configuration Constants
# =============================================================================

# Seed for reproducible random circuit generation
DEFAULT_RANDOM_SEED = 2236

# Standard gate set for random Clifford circuit generation
# Using only CX and H as per existing notebook patterns
CLIFFORD_GATE_SET = [CONTROLLED_X_GATE_NAME, HADAMARD_GATE_NAME]


# =============================================================================
# Coupling Map Fixtures (Using Qiskit Fake Backends)
# =============================================================================


@pytest.fixture
def lima_coupling_graph() -> nx.DiGraph:
    """Create coupling graph from IBM Lima backend (5 qubits, T-shape).

    Topology:
        0 - 1 - 2
            |
            3
            |
            4

    This is a small backend suitable for quick tests.

    Returns:
        NetworkX DiGraph from Qiskit FakeLimaV2 coupling map.
    """
    backend = qiskit_ibm_runtime.fake_provider.FakeLimaV2()
    return quariadne.milp.get_coupling_graph(backend.coupling_map)


@pytest.fixture
def manila_coupling_graph() -> nx.DiGraph:
    """Create coupling graph from IBM Manila backend (5 qubits, linear chain).

    Topology: 0 - 1 - 2 - 3 - 4

    Returns:
        NetworkX DiGraph from Qiskit FakeManilaV2 coupling map.
    """
    backend = qiskit_ibm_runtime.fake_provider.FakeManilaV2()
    return quariadne.milp.get_coupling_graph(backend.coupling_map)


@pytest.fixture
def ourense_coupling_graph() -> nx.DiGraph:
    """Create coupling graph from IBM Ourense backend (5 qubits).

    Returns:
        NetworkX DiGraph from Qiskit FakeOurenseV2 coupling map.
    """
    backend = qiskit_ibm_runtime.fake_provider.FakeOurenseV2()
    return quariadne.milp.get_coupling_graph(backend.coupling_map)


# =============================================================================
# Custom Coupling Map Fixtures (For Edge Case Testing)
# =============================================================================


@pytest.fixture
def linear_chain_3() -> nx.DiGraph:
    """Create a minimal 3-qubit linear chain coupling map.

    Topology: 0 <-> 1 <-> 2 (bidirectional)

    Returns:
        NetworkX DiGraph with 3 nodes and 4 directed edges.
    """
    coupling_map = nx.DiGraph()
    nodes = [quariadne.circuit.PhysicalQubit(i) for i in range(3)]
    coupling_map.add_nodes_from(nodes)

    # Bidirectional edges for linear chain
    edges = [
        (nodes[0], nodes[1]),
        (nodes[1], nodes[0]),
        (nodes[1], nodes[2]),
        (nodes[2], nodes[1]),
    ]
    coupling_map.add_edges_from(edges)
    return coupling_map


# =============================================================================
# Circuit Generation Functions
# =============================================================================


def create_qiskit_circuit_with_cx_gates(
    qubit_count: int,
    cx_pairs: list[tuple[int, int]],
) -> qiskit.QuantumCircuit:
    """Create a Qiskit QuantumCircuit with specified CX gate pairs.

    Args:
        qubit_count: Total number of qubits in the circuit
        cx_pairs: List of (control, target) qubit index pairs

    Returns:
        Qiskit QuantumCircuit with the specified CX gates.

    Example:
        >>> circuit = create_qiskit_circuit_with_cx_gates(3, [(0, 1), (1, 2)])
    """
    circuit = qiskit.QuantumCircuit(qubit_count)
    for control, target in cx_pairs:
        circuit.cx(control, target)
    return circuit


def convert_to_abstract_circuit(
    qiskit_circuit: qiskit.QuantumCircuit,
) -> quariadne.circuit.AbstractQuantumCircuit:
    """Convert Qiskit circuit to AbstractQuantumCircuit.

    Wrapper around AbstractQuantumCircuit.from_qiskit_circuit() for clarity.

    Args:
        qiskit_circuit: Qiskit QuantumCircuit to convert

    Returns:
        Quariadne AbstractQuantumCircuit representation.
    """
    return quariadne.circuit.AbstractQuantumCircuit.from_qiskit_circuit(qiskit_circuit)


# =============================================================================
# Pre-built Circuit Fixtures
# =============================================================================


@pytest.fixture
def trivial_circuit() -> quariadne.circuit.AbstractQuantumCircuit:
    """Create a trivial 2-qubit circuit with one CX gate.

    Circuit:
        q0: --*--
              |
        q1: --X--

    This circuit requires no swaps on any 2-qubit coupling map
    where q0 and q1 are adjacent.

    Returns:
        AbstractQuantumCircuit with 2 qubits and 1 CX operation.
    """
    qc = create_qiskit_circuit_with_cx_gates(2, [(0, 1)])
    return convert_to_abstract_circuit(qc)


@pytest.fixture
def linear_chain_circuit() -> quariadne.circuit.AbstractQuantumCircuit:
    """Create a circuit that may require swap on linear chain topology.

    Circuit:
        q0: --*---------
              |
        q1: --X----*----
                   |
        q2: -------X----

    On a 3-qubit linear chain (0-1-2), this can route without swaps
    if initial mapping is optimal.

    Returns:
        AbstractQuantumCircuit with 3 qubits and 2 CX operations.
    """
    qc = create_qiskit_circuit_with_cx_gates(3, [(0, 1), (1, 2)])
    return convert_to_abstract_circuit(qc)


@pytest.fixture
def sparse_first_layer_circuit() -> quariadne.circuit.AbstractQuantumCircuit:
    """Create a circuit where first layer doesn't use all qubits.

    Circuit (5 qubits, but first layer only uses q0, q1):
        q0: --*---------*--
              |         |
        q1: --X---------+--
                        |
        q2: -------*----X--
                   |
        q3: -------X-------
        q4: (unused)

    Tests initial mapping generation for unmapped qubits.

    Returns:
        AbstractQuantumCircuit with 5 qubits and 3 CX operations.
    """
    qc = create_qiskit_circuit_with_cx_gates(
        5,
        [
            (0, 1),  # Layer 1: only q0, q1 used
            (2, 3),  # Layer 2: q2, q3 used (parallel with different qubits)
            (0, 2),  # Layer 3: connects q0 to q2
        ],
    )
    return convert_to_abstract_circuit(qc)


@pytest.fixture
def random_clifford_circuit_small() -> quariadne.circuit.AbstractQuantumCircuit:
    """Create a small random Clifford circuit for testing.

    Uses Qiskit's random_clifford_circuit() generator with fixed seed
    for reproducibility.

    Returns:
        AbstractQuantumCircuit with 5 qubits and 20 random gates.
    """
    qc = qiskit.circuit.random.random_clifford_circuit(
        num_qubits=5,
        num_gates=20,
        gates=CLIFFORD_GATE_SET,
        seed=DEFAULT_RANDOM_SEED,
    )
    return convert_to_abstract_circuit(qc)
