"""Router comparison tests: Quariadne routers vs SABRE.

Compares routing algorithms via Qiskit's transpiler interface.

Ref: https://docs.pytest.org/en/stable/how-to/parametrize.html
Ref: https://docs.quantum.ibm.com/api/qiskit/transpiler
"""

import pytest
import qiskit
import qiskit.transpiler
import qiskit_ibm_runtime.fake_provider

from quariadne.benchmarks.constants import DEFAULT_OPTIMISATION_LEVEL, RoutingMethod
import quariadne.circuit
import quariadne.milp
import quariadne.routers
from tests.conftest import (
    CLIFFORD_GATE_SET,
    SWAP_GATE_NAME,
    create_qiskit_circuit_with_cx_gates,
)


# =============================================================================
# Test Constants
# =============================================================================

# Routing methods to test
METHODS_UNDER_TEST = (
    RoutingMethod.SABRE,
    RoutingMethod.QUARIADNE_LPE,
    RoutingMethod.QUARIADNE_LPM,
    RoutingMethod.QUARIADNE_ILP,
)

# Parametrize configuration for routing methods
METHOD_PARAM_NAME = "method"


def method_id(method: RoutingMethod) -> str:
    """Extract method name for pytest parametrize IDs."""
    return method.name


# Trivial circuit configuration: 2 qubits, single CX(0,1)
TRIVIAL_CIRCUIT_QUBIT_COUNT = 2
TRIVIAL_CIRCUIT_CX_PAIRS = [(0, 1)]
TRIVIAL_CIRCUIT_MAX_SWAPS = 1

# Linear chain circuit configuration: 3 qubits, CX(0,1) then CX(1,2)
LINEAR_CHAIN_QUBIT_COUNT = 3
LINEAR_CHAIN_CX_PAIRS = [(0, 1), (1, 2)]
LINEAR_CHAIN_MAX_SWAPS = 2

# Sparse circuit configuration: 5 qubits, first layer uses subset
SPARSE_CIRCUIT_QUBIT_COUNT = 5
SPARSE_CIRCUIT_CX_PAIRS = [(0, 1), (2, 3), (0, 2)]
SPARSE_CIRCUIT_MAX_SWAPS = 5

# Random Clifford circuit configuration
RANDOM_CLIFFORD_QUBIT_COUNT = 5
RANDOM_CLIFFORD_GATE_COUNT = 20
RANDOM_CLIFFORD_SEED = 2236

# Default count when gate type is absent from count_ops()
NO_GATES_COUNT = 0

# Number of qubits for two-qubit gates
TWO_QUBIT_GATE_ARITY = 2


def transpile_with_method(
    circuit: qiskit.QuantumCircuit,
    backend: qiskit.providers.BackendV2,
    method: RoutingMethod,
) -> qiskit.QuantumCircuit:
    """Transpile circuit using specified routing method.

    Args:
        circuit: Input quantum circuit to route.
        backend: Backend providing coupling map topology.
        method: Routing method from RoutingMethod enum.

    Returns:
        Routed quantum circuit with SWAPs inserted.
    """
    pm = qiskit.transpiler.generate_preset_pass_manager(
        backend=backend,
        optimization_level=DEFAULT_OPTIMISATION_LEVEL,
        routing_method=method.value,
        layout_method=method.layout_method,
    )
    return pm.run(circuit)


def all_two_qubit_gates_on_coupling_edges(
    circuit: qiskit.QuantumCircuit,
    coupling_map: qiskit.transpiler.CouplingMap,
) -> bool:
    """Check if all 2-qubit gates operate on connected qubits.

    Args:
        circuit: Routed quantum circuit to verify.
        coupling_map: Backend coupling map defining physical connectivity.

    Returns:
        True if all 2-qubit gates are on coupling map edges, False otherwise.
    """
    coupling_edges = set(coupling_map.get_edges())

    for instruction in circuit.data:
        if instruction.operation.num_qubits == TWO_QUBIT_GATE_ARITY:
            qubit_indices = tuple(circuit.find_bit(q).index for q in instruction.qubits)
            if qubit_indices not in coupling_edges:
                return False

    return True


class TestBasicRouting:
    """Basic routing tests for all methods."""

    @pytest.mark.parametrize(METHOD_PARAM_NAME, METHODS_UNDER_TEST, ids=method_id)
    def test_trivial_circuit(self, method: RoutingMethod) -> None:
        """Test trivial 2-qubit circuit routes successfully."""
        circuit = create_qiskit_circuit_with_cx_gates(
            TRIVIAL_CIRCUIT_QUBIT_COUNT,
            TRIVIAL_CIRCUIT_CX_PAIRS,
        )
        backend = qiskit_ibm_runtime.fake_provider.FakeLimaV2()

        routed = transpile_with_method(circuit, backend, method)

        assert routed is not None
        swap_count = routed.count_ops().get(SWAP_GATE_NAME, NO_GATES_COUNT)
        assert swap_count <= TRIVIAL_CIRCUIT_MAX_SWAPS

    @pytest.mark.parametrize(METHOD_PARAM_NAME, METHODS_UNDER_TEST, ids=method_id)
    def test_linear_chain_circuit(self, method: RoutingMethod) -> None:
        """Test linear chain circuit with sequential CX gates."""
        circuit = create_qiskit_circuit_with_cx_gates(
            LINEAR_CHAIN_QUBIT_COUNT,
            LINEAR_CHAIN_CX_PAIRS,
        )
        backend = qiskit_ibm_runtime.fake_provider.FakeLimaV2()

        routed = transpile_with_method(circuit, backend, method)

        assert routed is not None
        swap_count = routed.count_ops().get(SWAP_GATE_NAME, NO_GATES_COUNT)
        assert swap_count <= LINEAR_CHAIN_MAX_SWAPS

    @pytest.mark.parametrize(METHOD_PARAM_NAME, METHODS_UNDER_TEST, ids=method_id)
    def test_sparse_circuit(self, method: RoutingMethod) -> None:
        """Test sparse circuit where first layer uses subset of qubits."""
        circuit = create_qiskit_circuit_with_cx_gates(
            SPARSE_CIRCUIT_QUBIT_COUNT,
            SPARSE_CIRCUIT_CX_PAIRS,
        )
        backend = qiskit_ibm_runtime.fake_provider.FakeLimaV2()

        routed = transpile_with_method(circuit, backend, method)

        assert routed is not None
        swap_count = routed.count_ops().get(SWAP_GATE_NAME, NO_GATES_COUNT)
        assert swap_count <= SPARSE_CIRCUIT_MAX_SWAPS

    @pytest.mark.slow
    @pytest.mark.parametrize(METHOD_PARAM_NAME, METHODS_UNDER_TEST, ids=method_id)
    def test_random_clifford_circuit(self, method: RoutingMethod) -> None:
        """Test random Clifford circuit (stress test)."""
        circuit = qiskit.circuit.random.random_clifford_circuit(
            num_qubits=RANDOM_CLIFFORD_QUBIT_COUNT,
            num_gates=RANDOM_CLIFFORD_GATE_COUNT,
            gates=CLIFFORD_GATE_SET,
            seed=RANDOM_CLIFFORD_SEED,
        )
        backend = qiskit_ibm_runtime.fake_provider.FakeLimaV2()

        routed = transpile_with_method(circuit, backend, method)

        assert routed is not None
        assert routed.depth() > 0


class TestComparisonMetrics:
    """Record metrics for comparison report (use --html=report.html)."""

    @pytest.mark.parametrize(METHOD_PARAM_NAME, METHODS_UNDER_TEST, ids=method_id)
    def test_sparse_metrics(self, method: RoutingMethod, record_property) -> None:
        """Record metrics for sparse circuit."""
        circuit = create_qiskit_circuit_with_cx_gates(
            SPARSE_CIRCUIT_QUBIT_COUNT,
            SPARSE_CIRCUIT_CX_PAIRS,
        )
        backend = qiskit_ibm_runtime.fake_provider.FakeLimaV2()

        routed = transpile_with_method(circuit, backend, method)

        swap_count = routed.count_ops().get(SWAP_GATE_NAME, NO_GATES_COUNT)
        depth = routed.depth()

        record_property("swap_count", swap_count)
        record_property("depth", depth)

        assert routed is not None


# Methods with known bugs in consistency tests
METHODS_WITH_XFAIL_EXECUTABLE = (
    RoutingMethod.SABRE,
    RoutingMethod.QUARIADNE_LPE,
    pytest.param(
        RoutingMethod.QUARIADNE_LPM,
        marks=pytest.mark.xfail(reason="LPM produces non-adjacent CX gates"),
    ),
    RoutingMethod.QUARIADNE_ILP,
)


class TestRouterConsistency:
    """Verify routers produce valid, executable circuits."""

    @pytest.mark.parametrize(
        METHOD_PARAM_NAME, METHODS_WITH_XFAIL_EXECUTABLE, ids=method_id
    )
    def test_routed_circuits_are_executable(self, method: RoutingMethod) -> None:
        """Verify all 2-qubit gates operate on connected physical qubits."""
        circuit = create_qiskit_circuit_with_cx_gates(
            SPARSE_CIRCUIT_QUBIT_COUNT,
            SPARSE_CIRCUIT_CX_PAIRS,
        )
        backend = qiskit_ibm_runtime.fake_provider.FakeLimaV2()
        coupling_map = backend.coupling_map

        routed = transpile_with_method(circuit, backend, method)

        assert all_two_qubit_gates_on_coupling_edges(routed, coupling_map)

    @pytest.mark.parametrize(METHOD_PARAM_NAME, METHODS_UNDER_TEST, ids=method_id)
    def test_produces_bijective_mapping(self, method: RoutingMethod) -> None:
        """Verify initial layout is a valid bijection (no duplicate physical qubits)."""
        circuit = create_qiskit_circuit_with_cx_gates(
            SPARSE_CIRCUIT_QUBIT_COUNT,
            SPARSE_CIRCUIT_CX_PAIRS,
        )
        backend = qiskit_ibm_runtime.fake_provider.FakeLimaV2()

        routed = transpile_with_method(circuit, backend, method)

        initial_layout = routed.layout.initial_layout
        physical_qubits = list(initial_layout.get_physical_bits().keys())

        # Bijection: unique physical qubits for each logical qubit
        assert len(physical_qubits) == len(set(physical_qubits))


class TestDiagnosticSwapCounting:
    """Diagnostic tests to count SWAPs before transpiler decomposition."""

    def test_lpe_swap_count_before_decomposition(self) -> None:
        """Count SWAPs directly from LpRouterEdges before decomposition."""
        # Create circuit
        qiskit_circuit = create_qiskit_circuit_with_cx_gates(
            LINEAR_CHAIN_QUBIT_COUNT,
            LINEAR_CHAIN_CX_PAIRS,
        )
        quariadne_circuit = (
            quariadne.circuit.AbstractQuantumCircuit.from_qiskit_circuit(qiskit_circuit)
        )

        # Get coupling map
        backend = qiskit_ibm_runtime.fake_provider.FakeLimaV2()
        coupling_graph = quariadne.milp.get_coupling_graph(backend.coupling_map)

        # Run router directly (no transpiler decomposition)
        router = quariadne.routers.LpRouterEdges(coupling_graph, quariadne_circuit)
        inserted_swaps = router.get_inserted_swaps()

        # Count total SWAPs
        total_swaps = sum(len(swaps) for swaps in inserted_swaps.values())

        print(f"\nLPE Router: {total_swaps} SWAPs inserted")
        print(f"  Swaps by layer: {dict(inserted_swaps)}")

        # This test is diagnostic - just verify router runs
        assert inserted_swaps is not None
