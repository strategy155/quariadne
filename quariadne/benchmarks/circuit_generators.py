"""Circuit generation functions for scaling benchmarks.

This module provides functions to generate quantum circuits for benchmarking
routing algorithms. Supports random Clifford circuits and structured patterns.

New generators for comprehensive benchmarking:
    - QFT-like circuits (all-to-all entanglement pattern)
    - Full-layer circuits (dense parallel CX layers, Pozzi 2020)
    - Depth-controlled random circuits (Maslov 2008)

References:
    - Qiskit random circuits: https://docs.quantum.ibm.com/api/qiskit/circuit
    - Python 3.13 random module: https://docs.python.org/3/library/random.html
    - Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
    - Pozzi et al. 2020: "Using Reinforcement Learning to Perform Qubit Routing"
    - Maslov 2008: "Linear depth stabilizer and quantum Fourier transformation"
    - Nielsen & Chuang: Quantum Computation and Quantum Information, Ch. 5
"""

import random

import qiskit
import qiskit.circuit.random

import quariadne.circuit


# Default gate set for random Clifford circuits (CX for entanglement, H for superposition)
CLIFFORD_GATE_SET = ["cx", "h"]


def generate_random_clifford_circuit(
    num_qubits: int,
    num_gates: int,
    seed: int,
) -> quariadne.circuit.AbstractQuantumCircuit:
    """Generate a random Clifford circuit with CX and H gates.

    Creates a random circuit using Qiskit's random_clifford_circuit function,
    then converts it to the internal AbstractQuantumCircuit representation.
    The circuit uses only CX (entangling) and H (single-qubit) gates.

    Args:
        num_qubits: Number of qubits in the circuit. Must be positive.
        num_gates: Total number of gates to include. Must be non-negative.
        seed: Random seed for reproducibility.

    Returns:
        AbstractQuantumCircuit ready for routing.

    Raises:
        ValueError: If num_qubits < 1 or num_gates < 0.

    Example:
        >>> circuit = generate_random_clifford_circuit(5, 20, seed=42)
        >>> len(circuit.operations)
        20
    """
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be positive, got {num_qubits}")
    if num_gates < 0:
        raise ValueError(f"num_gates must be non-negative, got {num_gates}")

    qiskit_circuit = qiskit.circuit.random.random_clifford_circuit(
        num_qubits=num_qubits,
        num_gates=num_gates,
        gates=CLIFFORD_GATE_SET,
        seed=seed,
    )
    return quariadne.circuit.AbstractQuantumCircuit.from_qiskit_circuit(qiskit_circuit)


def generate_linear_chain_circuit(
    num_qubits: int,
    chain_length: int,
) -> quariadne.circuit.AbstractQuantumCircuit:
    """Generate a linear chain of CX gates: (0,1), (1,2), (2,3), ...

    Creates a circuit where each qubit is entangled with its neighbour in
    sequence. This pattern is challenging on non-linear topologies like Ourense
    where not all consecutive qubits are directly connected.

    Args:
        num_qubits: Number of qubits in the circuit. Must be at least 2.
        chain_length: Number of CX gates in the chain. Must be non-negative.

    Returns:
        AbstractQuantumCircuit with linear chain pattern.

    Raises:
        ValueError: If num_qubits < 2 or chain_length < 0.

    Example:
        >>> circuit = generate_linear_chain_circuit(4, 6)
        >>> # Creates: CX(0,1), CX(1,2), CX(2,3), CX(0,1), CX(1,2), CX(2,3)
    """
    if num_qubits < 2:
        raise ValueError(f"num_qubits must be at least 2, got {num_qubits}")
    if chain_length < 0:
        raise ValueError(f"chain_length must be non-negative, got {chain_length}")

    qiskit_circuit = qiskit.QuantumCircuit(num_qubits)
    for i in range(chain_length):
        control = i % (num_qubits - 1)
        target = control + 1
        qiskit_circuit.cx(control, target)

    return quariadne.circuit.AbstractQuantumCircuit.from_qiskit_circuit(qiskit_circuit)


def generate_ring_circuit(
    num_qubits: int,
) -> quariadne.circuit.AbstractQuantumCircuit:
    """Generate a ring circuit: (0,1), (1,2), ..., (n-2,n-1), (n-1,0).

    Creates a circuit forming a complete cycle where the last qubit connects
    back to the first. This pattern requires routing for any non-ring topology.

    Args:
        num_qubits: Number of qubits in the circuit. Must be at least 2.

    Returns:
        AbstractQuantumCircuit with ring pattern (n CX gates for n qubits).

    Raises:
        ValueError: If num_qubits < 2.

    Example:
        >>> circuit = generate_ring_circuit(4)
        >>> # Creates: CX(0,1), CX(1,2), CX(2,3), CX(3,0)
    """
    if num_qubits < 2:
        raise ValueError(f"num_qubits must be at least 2, got {num_qubits}")

    qiskit_circuit = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        control = i
        target = (i + 1) % num_qubits
        qiskit_circuit.cx(control, target)

    return quariadne.circuit.AbstractQuantumCircuit.from_qiskit_circuit(qiskit_circuit)


def generate_star_circuit(
    num_qubits: int,
    hub_qubit: int = 0,
) -> quariadne.circuit.AbstractQuantumCircuit:
    """Generate a star circuit where one hub qubit connects to all others.

    Creates a circuit with the hub qubit as control for CX gates to every
    other qubit. Tests routers with concentrated connectivity around one qubit.
    On Ourense topology, qubit 1 is the natural hub (degree 3).

    Args:
        num_qubits: Number of qubits in the circuit. Must be at least 2.
        hub_qubit: Index of the central hub qubit. Defaults to 0.

    Returns:
        AbstractQuantumCircuit with star pattern (n-1 CX gates for n qubits).

    Raises:
        ValueError: If num_qubits < 2 or hub_qubit is out of range.

    Example:
        >>> circuit = generate_star_circuit(4, hub_qubit=0)
        >>> # Creates: CX(0,1), CX(0,2), CX(0,3)
    """
    if num_qubits < 2:
        raise ValueError(f"num_qubits must be at least 2, got {num_qubits}")
    if hub_qubit < 0 or hub_qubit >= num_qubits:
        raise ValueError(
            f"hub_qubit must be in range [0, {num_qubits}), got {hub_qubit}"
        )

    qiskit_circuit = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        if i != hub_qubit:
            qiskit_circuit.cx(hub_qubit, i)

    return quariadne.circuit.AbstractQuantumCircuit.from_qiskit_circuit(qiskit_circuit)


def generate_qft_like_circuit(
    num_qubits: int,
    seed: int | None = None,
) -> quariadne.circuit.AbstractQuantumCircuit:
    """Generate a QFT-inspired circuit with all-to-all entanglement pattern.

    Creates a circuit mimicking the Quantum Fourier Transform structure:
    CX gates between all pairs (i, j) where i < j. This produces a dense
    entanglement pattern that challenges routers on sparse topologies.

    The QFT pattern requires O(n^2) two-qubit gates for n qubits, making it
    a stress test for routing algorithms on non-fully-connected hardware.

    Args:
        num_qubits: Number of qubits in the circuit. Must be at least 2.
        seed: Random seed for optional gate ordering permutation. If None,
            gates are applied in canonical order (i < j, sorted by i then j).

    Returns:
        AbstractQuantumCircuit with QFT-like entanglement pattern.
        Total gates: n*(n-1)/2 CX gates for n qubits.

    Raises:
        ValueError: If num_qubits < 2.

    Example:
        >>> circuit = generate_qft_like_circuit(4)
        >>> # Creates: CX(0,1), CX(0,2), CX(0,3), CX(1,2), CX(1,3), CX(2,3)
        >>> len(circuit.get_two_qubit_operations())
        6

    Reference:
        - Nielsen & Chuang, Quantum Computation and Quantum Information, Ch. 5
    """
    if num_qubits < 2:
        raise ValueError(f"num_qubits must be at least 2, got {num_qubits}")

    # Generate all pairs (i, j) where i < j for complete entanglement graph
    all_qubit_pairs = [
        (control_qubit, target_qubit)
        for control_qubit in range(num_qubits)
        for target_qubit in range(control_qubit + 1, num_qubits)
    ]

    # Optionally shuffle pairs for varied dependency patterns
    if seed is not None:
        random_generator = random.Random(seed)
        random_generator.shuffle(all_qubit_pairs)

    qiskit_circuit = qiskit.QuantumCircuit(num_qubits)
    for control_qubit, target_qubit in all_qubit_pairs:
        qiskit_circuit.cx(control_qubit, target_qubit)

    return quariadne.circuit.AbstractQuantumCircuit.from_qiskit_circuit(qiskit_circuit)


def generate_full_layer_circuit(
    num_qubits: int,
    num_layers: int,
    seed: int,
) -> quariadne.circuit.AbstractQuantumCircuit:
    """Generate a circuit with dense parallel CX layers.

    Creates layers where maximum parallel CX gates execute simultaneously.
    Each layer pairs qubits in two alternating patterns:
        - Even layers: (0,1), (2,3), (4,5), ...
        - Odd layers: (1,2), (3,4), (5,6), ...

    This pattern was used in Pozzi et al. 2020 for benchmarking reinforcement
    learning-based qubit routing.

    Args:
        num_qubits: Number of qubits in the circuit. Must be at least 2.
        num_layers: Number of full layers to generate. Must be non-negative.
        seed: Random seed (reserved for future layer permutations).

    Returns:
        AbstractQuantumCircuit with dense parallel layers.
        Total gates: ~(num_qubits // 2) * num_layers CX gates.

    Raises:
        ValueError: If num_qubits < 2 or num_layers < 0.

    Example:
        >>> circuit = generate_full_layer_circuit(4, 2, seed=42)
        >>> # Layer 0 (even): CX(0,1), CX(2,3)
        >>> # Layer 1 (odd): CX(1,2)

    Reference:
        - Pozzi et al. 2020: "Using Reinforcement Learning to Perform
          Qubit Routing in Quantum Compilers"
    """
    if num_qubits < 2:
        raise ValueError(f"num_qubits must be at least 2, got {num_qubits}")
    if num_layers < 0:
        raise ValueError(f"num_layers must be non-negative, got {num_layers}")

    # Seed reserved for future use (e.g. layer permutations)
    _ = seed

    qiskit_circuit = qiskit.QuantumCircuit(num_qubits)

    for layer_index in range(num_layers):
        # Determine starting qubit for this layer's pairing
        if layer_index % 2 == 0:
            # Even layers: pair (0,1), (2,3), (4,5), ...
            pairing_start_qubit = 0
        else:
            # Odd layers: pair (1,2), (3,4), (5,6), ...
            pairing_start_qubit = 1

        for qubit_index in range(pairing_start_qubit, num_qubits - 1, 2):
            control_qubit = qubit_index
            target_qubit = qubit_index + 1
            qiskit_circuit.cx(control_qubit, target_qubit)

    return quariadne.circuit.AbstractQuantumCircuit.from_qiskit_circuit(qiskit_circuit)


def generate_depth_controlled_random_circuit(
    num_qubits: int,
    target_depth: int,
    seed: int,
) -> quariadne.circuit.AbstractQuantumCircuit:
    """Generate a random circuit with controlled depth.

    Creates a random circuit where the circuit depth (longest path through
    the dependency graph) is approximately equal to target_depth. Uses a
    layer-by-layer generation approach where each layer adds parallel gates.

    This approach follows Maslov 2008's composable subcircuit methodology,
    where random patterns are generated in layers for cacheable generation.

    Args:
        num_qubits: Number of qubits in the circuit. Must be at least 2.
        target_depth: Target circuit depth (approximate). Must be positive.
        seed: Random seed for reproducibility.

    Returns:
        AbstractQuantumCircuit with controlled depth.
        Actual depth may vary slightly from target due to random gate placement.

    Raises:
        ValueError: If num_qubits < 2 or target_depth < 1.

    Example:
        >>> circuit = generate_depth_controlled_random_circuit(8, 10, seed=42)
        >>> # Creates ~10 layers of random CX gates

    Reference:
        - Maslov 2008: "Linear depth stabilizer and quantum Fourier
          transformation circuits with no auxiliary qubits"
    """
    if num_qubits < 2:
        raise ValueError(f"num_qubits must be at least 2, got {num_qubits}")
    if target_depth < 1:
        raise ValueError(f"target_depth must be positive, got {target_depth}")

    random_generator = random.Random(seed)
    qiskit_circuit = qiskit.QuantumCircuit(num_qubits)

    for _ in range(target_depth):
        # Each layer: randomly select non-overlapping qubit pairs
        available_qubit_indices = list(range(num_qubits))
        random_generator.shuffle(available_qubit_indices)

        # Create pairs from shuffled qubits
        num_pairs_in_layer = len(available_qubit_indices) // 2
        for pair_index in range(num_pairs_in_layer):
            control_qubit = available_qubit_indices[2 * pair_index]
            target_qubit = available_qubit_indices[2 * pair_index + 1]
            qiskit_circuit.cx(control_qubit, target_qubit)

    return quariadne.circuit.AbstractQuantumCircuit.from_qiskit_circuit(qiskit_circuit)
