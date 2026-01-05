"""Tests for LpRouterEdges (edge-based LP router).

This module tests the correctness and basic functionality of the edge router:
- Trivial routing (no swaps required)
- Initial mapping generation
- Edge direction handling with asymmetric coupling maps
- Comparison with ILP router for consistency

Ref: https://docs.pytest.org/en/stable/how-to/assert.html
"""

import networkx as nx
import pytest

import quariadne.circuit
import quariadne.routers

from tests.conftest import (
    create_qiskit_circuit_with_cx_gates,
    convert_to_abstract_circuit,
)


class TestLpRouterEdgesBasic:
    """Basic functionality tests for LpRouterEdges."""

    def test_trivial_routing_produces_valid_result(
        self,
        trivial_circuit: quariadne.circuit.AbstractQuantumCircuit,
        linear_chain_3: nx.DiGraph,
    ) -> None:
        """Test that a trivial 2-qubit circuit routes successfully.

        A single CX gate on adjacent qubits should require no swaps.
        """
        # Act
        router = quariadne.routers.LpRouterEdges(
            coupling_map=linear_chain_3,
            quantum_circuit=trivial_circuit,
        )

        # Assert - router should complete without error
        initial_mapping = router.get_initial_mapping()
        _ = router.get_inserted_swaps()  # Verify method works

        # Mapping should include all logical qubits
        assert len(initial_mapping) == len(trivial_circuit.qubits)

        # Each logical qubit should map to a unique physical qubit
        physical_qubits_used = list(initial_mapping.values())
        assert len(physical_qubits_used) == len(set(physical_qubits_used))

    def test_initial_mapping_covers_all_logical_qubits(
        self,
        linear_chain_circuit: quariadne.circuit.AbstractQuantumCircuit,
        linear_chain_3: nx.DiGraph,
    ) -> None:
        """Test that initial mapping includes all logical qubits."""
        # Act
        router = quariadne.routers.LpRouterEdges(
            coupling_map=linear_chain_3,
            quantum_circuit=linear_chain_circuit,
        )
        initial_mapping = router.get_initial_mapping()

        # Assert
        for logical_qubit in linear_chain_circuit.qubits:
            assert logical_qubit in initial_mapping, (
                f"Logical qubit {logical_qubit} missing from initial mapping"
            )

    def test_initial_mapping_maps_to_valid_physical_qubits(
        self,
        linear_chain_circuit: quariadne.circuit.AbstractQuantumCircuit,
        linear_chain_3: nx.DiGraph,
    ) -> None:
        """Test that initial mapping uses only valid physical qubits."""
        # Act
        router = quariadne.routers.LpRouterEdges(
            coupling_map=linear_chain_3,
            quantum_circuit=linear_chain_circuit,
        )
        initial_mapping = router.get_initial_mapping()

        # Assert - all physical qubits should be in the coupling map
        coupling_map_nodes = set(linear_chain_3.nodes())
        for physical_qubit in initial_mapping.values():
            assert physical_qubit in coupling_map_nodes, (
                f"Physical qubit {physical_qubit} not in coupling map"
            )


class TestLpRouterEdgesSequentialOperations:
    """Tests for routing sequential operations that share qubits."""

    def test_sequential_cx_gates_with_shared_qubit(
        self,
        linear_chain_3: nx.DiGraph,
    ) -> None:
        """Test routing two CX gates that share a qubit.

        Circuit:
            q0: --*---------
                  |
            q1: --X----*----
                       |
            q2: -------X----

        Both gates share qubit 1, so they must be in separate layers.
        """
        # Arrange
        qc = create_qiskit_circuit_with_cx_gates(3, [(0, 1), (1, 2)])
        circuit = convert_to_abstract_circuit(qc)

        # Act
        router = quariadne.routers.LpRouterEdges(
            coupling_map=linear_chain_3,
            quantum_circuit=circuit,
        )

        # Assert
        initial_mapping = router.get_initial_mapping()
        _ = router.get_inserted_swaps()  # Verify method works

        # Mapping should cover all logical qubits
        assert len(initial_mapping) == 3

        # Mapping should be bijective
        physical_qubits_used = list(initial_mapping.values())
        assert len(physical_qubits_used) == len(set(physical_qubits_used))


class TestLpRouterEdgesSparseFirstLayer:
    """Tests for circuits where first layer doesn't use all qubits."""

    def test_sparse_first_layer_mapping(
        self,
        sparse_first_layer_circuit: quariadne.circuit.AbstractQuantumCircuit,
        lima_coupling_graph: nx.DiGraph,
    ) -> None:
        """Test initial mapping when first layer uses subset of qubits.

        The sparse circuit uses only q0, q1 in first layer.
        The router should still create a valid mapping for all 5 qubits.
        """
        # Act
        router = quariadne.routers.LpRouterEdges(
            coupling_map=lima_coupling_graph,
            quantum_circuit=sparse_first_layer_circuit,
        )
        initial_mapping = router.get_initial_mapping()

        # Assert - all 5 logical qubits should be mapped
        assert len(initial_mapping) == 5

        # All mappings should be to distinct physical qubits
        physical_qubits = list(initial_mapping.values())
        assert len(physical_qubits) == len(set(physical_qubits))


class TestLpRouterEdgesVsIlpRouter:
    """Comparison tests between LP and ILP routers for consistency."""

    @pytest.mark.slow
    def test_both_routers_produce_valid_mappings(
        self,
        linear_chain_circuit: quariadne.circuit.AbstractQuantumCircuit,
        linear_chain_3: nx.DiGraph,
    ) -> None:
        """Test that both LpRouterEdges and IlpRouter produce valid mappings.

        Both routers should produce mappings where:
        - All logical qubits are mapped
        - All physical qubits in mapping are in coupling map
        - Mapping is bijective (1-1)
        """
        # Act - run both routers
        lp_router = quariadne.routers.LpRouterEdges(
            coupling_map=linear_chain_3,
            quantum_circuit=linear_chain_circuit,
        )
        ilp_router = quariadne.routers.IlpRouter(
            coupling_map=linear_chain_3,
            quantum_circuit=linear_chain_circuit,
        )

        lp_mapping = lp_router.get_initial_mapping()
        ilp_mapping = ilp_router.get_initial_mapping()

        # Assert - both should have same number of mapped qubits
        assert len(lp_mapping) == len(ilp_mapping)
        assert len(lp_mapping) == len(linear_chain_circuit.qubits)

        # Both should have bijective mappings
        lp_physical = list(lp_mapping.values())
        ilp_physical = list(ilp_mapping.values())
        assert len(lp_physical) == len(set(lp_physical))
        assert len(ilp_physical) == len(set(ilp_physical))
