"""Tests for BipartiteAllocationRouter.

This module tests the LP-based bipartite allocation router:
- Basic routing functionality and result structure
- Constraint satisfaction (operation uniqueness, edge validity)
- Swap extraction correctness
- End-to-end routing validation through Qiskit transpiler

Ref: https://docs.pytest.org/en/stable/how-to/assert.html
"""

import networkx as nx
import pytest
import qiskit.transpiler
from qiskit.converters import circuit_to_dag

import quariadne.benchmarks.circuit_generators as cg
import quariadne.benchmarks.topology_generators as tg
import quariadne.circuit
from quariadne.bipartite_allocation_lp import BipartiteAllocationRouter


class TestBipartiteAllocationRouterBasic:
    """Basic functionality tests for BipartiteAllocationRouter."""

    def test_trivial_routing_succeeds(
        self,
        trivial_circuit: quariadne.circuit.AbstractQuantumCircuit,
        linear_chain_3: nx.DiGraph,
    ) -> None:
        """Test that a trivial 2-qubit circuit routes successfully.

        A single CX gate on adjacent qubits should complete without error
        and produce a non-negative objective value.
        """
        router = BipartiteAllocationRouter(
            coupling_map=linear_chain_3,
            quantum_circuit=trivial_circuit,
        )
        result = router.run()

        assert result.objective_value >= 0


class TestBipartiteRoutingValidity:
    """Validate that BipartiteRouting produces valid circuits.

    These tests verify that the end-to-end transpilation pipeline produces
    circuits where all two-qubit gates operate on adjacent physical qubits.
    This catches ordering mismatches between the LP solver and routing pass.

    Ref: https://docs.pytest.org/en/stable/how-to/parametrize.html
    """

    @pytest.mark.parametrize("seed", [42, 48, 123])
    def test_qft_like_on_watts_strogatz_produces_valid_routing(self, seed: int) -> None:
        """Test that QFT-like circuits route correctly on Watts-Strogatz topology.

        This is the reproduction case from the bug report: all-to-all CX
        connectivity on a sparse topology should produce valid routing with
        all CX gates on adjacent qubits.

        Args:
            seed: Random seed for circuit and topology generation.
        """
        num_qubits = 6
        circuit = cg.generate_qft_like_circuit(num_qubits, seed=seed)
        coupling_map = tg.connected_watts_strogatz_to_coupling_map(
            num_qubits=num_qubits,
            nearest_neighbours=2,
            rewiring_probability=0.3,
            seed=seed,
        )

        pm = qiskit.transpiler.generate_preset_pass_manager(
            coupling_map=coupling_map,
            optimization_level=0,
            routing_method="quariadne_bipartite",
            layout_method="quariadne_bipartite",
        )
        transpiled = pm.run(circuit.to_qiskit())

        # Build valid edge set (normalised to allow bidirectional lookup)
        valid_edges = set((min(a, b), max(a, b)) for a, b in coupling_map.get_edges())
        dag = circuit_to_dag(transpiled)

        # Count invalid CX gates (those on non-adjacent qubits)
        invalid_count = 0
        total_cx_count = 0
        for node in dag.op_nodes():
            if node.op.name == "cx":
                total_cx_count += 1
                p0 = transpiled.find_bit(node.qargs[0]).index
                p1 = transpiled.find_bit(node.qargs[1]).index
                edge_key = (min(p0, p1), max(p0, p1))
                if edge_key not in valid_edges:
                    invalid_count += 1

        assert invalid_count == 0, (
            f"Invalid CX gates: {invalid_count} / {total_cx_count}"
        )

    def test_linear_chain_circuit_produces_valid_routing(self) -> None:
        """Test that a simple linear chain circuit routes correctly.

        Linear chain circuits are a basic stress test for routing passes.
        """
        num_qubits = 5
        circuit = cg.generate_linear_chain_circuit(num_qubits, chain_length=8)
        coupling_map = tg.connected_watts_strogatz_to_coupling_map(
            num_qubits=num_qubits,
            nearest_neighbours=2,
            rewiring_probability=0.2,
            seed=42,
        )

        pm = qiskit.transpiler.generate_preset_pass_manager(
            coupling_map=coupling_map,
            optimization_level=0,
            routing_method="quariadne_bipartite",
            layout_method="quariadne_bipartite",
        )
        transpiled = pm.run(circuit.to_qiskit())

        # Validate all CX gates are on valid edges
        valid_edges = set((min(a, b), max(a, b)) for a, b in coupling_map.get_edges())
        dag = circuit_to_dag(transpiled)

        for node in dag.op_nodes():
            if node.op.name == "cx":
                p0 = transpiled.find_bit(node.qargs[0]).index
                p1 = transpiled.find_bit(node.qargs[1]).index
                edge_key = (min(p0, p1), max(p0, p1))
                assert edge_key in valid_edges, f"CX on invalid edge ({p0}, {p1})"
