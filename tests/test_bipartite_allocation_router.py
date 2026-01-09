"""Tests for BipartiteAllocationRouter.

This module tests the LP-based bipartite allocation router:
- Basic routing functionality and result structure
- Constraint satisfaction (operation uniqueness, edge validity)
- Swap extraction correctness

Ref: https://docs.pytest.org/en/stable/how-to/assert.html
"""

import networkx as nx

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
