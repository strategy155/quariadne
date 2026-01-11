"""Unified LP Router for quantum circuit routing.

This module implements a Linear Programming formulation for quantum circuit routing
using unified operation pair transitions. Unlike BipartiteAllocationRouter which
creates one z variable per qubit flowing between operations, this router creates
one z variable per operation pair regardless of how many qubits they share.

Mathematical Formulation
------------------------
**Decision Variables:**
    - ``y_{o,e} ∈ [0,1]``: Operation o assigned to edge e.
    - ``z_{o₁→o₂, e₁, e₂} ∈ [0,1]``: Flow from edge e₁ at o₁ to edge e₂ at o₂.
      One variable per operation pair (not per qubit).

**Constraints:**
    1. Operation Assignment: ``Σ_e y_{o,e} = 1``
    2. Position Exclusivity: ``Σ_{o ∈ layer_t, e : p ∈ e} y_{o,e} ≤ 1``
    3. Flow Conservation: outflow and inflow constraints linking y and z.

**Objective:**
    Minimise total distance, where cost sums distances for all shared qubits.

References
----------
- HiGHS Python Interface: https://ergo-code.github.io/HiGHS/dev/interfaces/python/
- Python dataclasses: https://docs.python.org/3/library/dataclasses.html
"""

from __future__ import annotations

import copy
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import highspy
import networkx as nx
import numpy as np

import quariadne.circuit
from quariadne.lp_router_base import (
    BaseLPRouter,
    LPSolverOptions,
    PhysicalEdge,
    COEFFICIENT_POSITIVE,
    COEFFICIENT_NEGATIVE,
    EQUALITY_BOUND_ZERO,
    Y_VARIABLE_LOWER_BOUND,
    Y_VARIABLE_UPPER_BOUND,
    Z_VARIABLE_LOWER_BOUND,
    CONSTRAINT_ARRAY_DTYPE,
    INDEX_ARRAY_DTYPE,
    compute_all_pairs_shortest_paths,
)

_logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class OperationPairTransition:
    """Unified transition between consecutive operation pairs.

    Unlike FlowTransition (one per qubit), this creates one transition per
    operation pair regardless of how many qubits they share. This reduces
    the number of z variables when operations share multiple qubits.

    The distance cost for this transition is the sum of distances for all
    shared qubits, correctly penalising total movement.

    Attributes:
        from_operation_index: Index of the source operation.
        to_operation_index: Index of the target operation.
        shared_qubits: Tuple of logical qubits that both operations share.

    Reference:
        - https://docs.python.org/3/library/dataclasses.html#frozen-instances
    """

    from_operation_index: int
    to_operation_index: int
    shared_qubits: tuple[quariadne.circuit.LogicalQubit, ...]


class UnifiedVariableType(Enum):
    """Enum for variable types in unified LP router.

    Attributes:
        Y_OPERATION_EDGE: Variable y_{o,e} for operation-edge assignment.
        Z_FLOW: Flow variable z_{o₁→o₂, e₁, e₂} for operation pair transitions.
    """

    Y_OPERATION_EDGE = "y_operation_edge"
    Z_FLOW = "z_flow"


@dataclass
class UnifiedLPResult:
    """Result container for unified LP router optimisation.

    Contains the solution mapping operations to edges, the derived initial
    qubit mapping, and the swap sequence extracted from z flow variables.

    Attributes:
        objective_value: The optimal objective value (total summed distance).
        operation_edge_assignment: Dict mapping operation index to assigned edge.
        operation_to_edge_by_qubits: Dict mapping (qubit1, qubit2, occurrence) to edge.
        initial_mapping: Initial logical-to-physical qubit mapping.
        final_mapping: Qubit mapping after all operations.
        operations: List of operations in LP order for reference.
        inserted_swaps: Dict mapping operation index to list of PhysicalSwap objects.
    """

    objective_value: float
    operation_edge_assignment: dict[
        int, tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit]
    ]
    operation_to_edge_by_qubits: dict[
        tuple[int, int, int],
        tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit],
    ]
    initial_mapping: dict[
        quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit
    ]
    final_mapping: dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]
    operations: list[quariadne.circuit.QuantumOperation]
    inserted_swaps: dict[int, list[quariadne.circuit.PhysicalSwap]]


# =============================================================================
# Main Router Class
# =============================================================================


class LpRouterUnified(BaseLPRouter):
    """LP router with unified operation pair transitions.

    This router implements the edge-flow formulation with unified z variables
    per operation pair. When two consecutive operations share multiple qubits,
    they share a single z variable set (instead of one per qubit).

    The objective function sums distances for all shared qubits, correctly
    penalising total qubit movement.

    Attributes:
        coupling_map: NetworkX DiGraph representing hardware connectivity.
        routed_circuit: The quantum circuit to route.
        solver_options: Configuration for HiGHS solver.
    """

    def __init__(
        self,
        coupling_map: nx.DiGraph[quariadne.circuit.PhysicalQubit],
        quantum_circuit: quariadne.circuit.AbstractQuantumCircuit,
        solver_options: LPSolverOptions | None = None,
        fixed_operation_edges: dict[int, int] | None = None,
    ) -> None:
        """Initialise the unified LP router."""
        super().__init__(
            coupling_map=coupling_map,
            quantum_circuit=quantum_circuit,
            solver_options=solver_options,
            fixed_operation_edges=fixed_operation_edges,
        )

        # Build unified transitions (specific to this router)
        self.transitions = self._build_operation_pair_transitions()
        self.transition_count = len(self.transitions)

        # Set up variable indexing (requires transition_count)
        self._setup_variable_indexing()

    def _preprocess_circuit(
        self, quantum_circuit: quariadne.circuit.AbstractQuantumCircuit
    ) -> quariadne.circuit.AbstractQuantumCircuit:
        """Preprocess circuit: deepcopy and add dummy qubits if needed."""
        circuit = copy.deepcopy(quantum_circuit)

        qubit_count = self.coupling_map.number_of_nodes()
        circuit_qubit_count = len(circuit.qubits)

        if circuit_qubit_count < qubit_count:
            dummy_qubits = tuple(
                quariadne.circuit.LogicalQubit(i)
                for i in range(circuit_qubit_count, qubit_count)
            )
            circuit.qubits = circuit.qubits + dummy_qubits

        return circuit

    def _get_operations(self) -> list[quariadne.circuit.QuantumOperation]:
        """Get all operations from the circuit."""
        return list(self.routed_circuit.operations)

    def _setup_variable_indexing(self) -> None:
        """Set up variable indexing for unified formulation."""
        self.y_var_count = self.operation_count * self.edge_count
        self.y_var_offset = 0

        # z indexed by operation pair (not by qubit)
        self.z_var_count = self.transition_count * self.edge_count * self.edge_count
        self.z_var_offset = self.y_var_count

        self.total_var_count = self.y_var_count + self.z_var_count

    def _build_operation_pair_transitions(self) -> list[OperationPairTransition]:
        """Build unique transitions per operation pair.

        For each pair of consecutive operations that share at least one qubit,
        creates a single OperationPairTransition containing all shared qubits.
        This differs from BipartiteAllocationRouter which creates one transition
        per qubit.

        Returns:
            List of OperationPairTransition objects, sorted by (from_op, to_op).
        """
        # Map each (from_operation, to_operation) pair to the qubits they share
        pair_to_shared_qubits: dict[
            tuple[int, int], list[quariadne.circuit.LogicalQubit]
        ] = defaultdict(list)

        # Iterate over each qubit's operation sequence
        for qubit, operation_info in self.qubit_operations.items():
            operation_indices = operation_info.operation_indices

            # Create transitions between consecutive operations for this qubit
            for i in range(1, len(operation_indices)):
                from_operation = operation_indices[i - 1]
                to_operation = operation_indices[i]
                pair_to_shared_qubits[(from_operation, to_operation)].append(qubit)

        # Build transition objects, sorted for deterministic ordering
        transitions = [
            OperationPairTransition(
                from_operation_index=from_operation,
                to_operation_index=to_operation,
                shared_qubits=tuple(shared_qubits),
            )
            for (from_operation, to_operation), shared_qubits in sorted(
                pair_to_shared_qubits.items()
            )
        ]

        return transitions

    def _get_z_var_index(
        self,
        transition_index: int,
        from_edge_index: int,
        to_edge_index: int,
    ) -> int:
        """Get the variable index for a z flow variable.

        The z variables are laid out in a 3D structure:
        [transition_0][from_edge_0][to_edge_0..E-1], [transition_0][from_edge_1][...], ...

        Args:
            transition_index: Index of the operation pair transition.
            from_edge_index: Index of the source edge.
            to_edge_index: Index of the target edge.

        Returns:
            The flattened variable index in the LP formulation.
        """
        # Calculate position within the z variable block
        local_index = (
            transition_index * self.edge_count * self.edge_count
            + from_edge_index * self.edge_count
            + to_edge_index
        )
        return self.z_var_offset + local_index

    def _add_flow_conservation_constraints(self, highs_model: highspy.Highs) -> None:
        """Add outflow-only flow conservation constraints.

        For each transition and each source edge, ensures that the total flow
        leaving that edge equals the assignment of the source operation to
        that edge.

        Uses only outflow constraints (not both outflow and inflow) to avoid
        rank deficiency in the constraint matrix. Mathematical proof shows
        that outflow and inflow constraints are linearly dependent.

        Constraint form:
            Σ_{to_edge} z_{transition, from_edge, to_edge} = y_{from_op, from_edge}

        Args:
            highs_model: The HiGHS model instance to add constraints to.
        """
        # Calculate constraint dimensions
        num_constraints = self.transition_count * self.edge_count
        nonzeros_per_constraint = self.edge_count + 1  # E z-variables + 1 y-variable
        total_nonzeros = num_constraints * nonzeros_per_constraint

        # Pre-allocate CSR format arrays
        lower_bounds = np.full(
            num_constraints, EQUALITY_BOUND_ZERO, dtype=CONSTRAINT_ARRAY_DTYPE
        )
        upper_bounds = np.full(
            num_constraints, EQUALITY_BOUND_ZERO, dtype=CONSTRAINT_ARRAY_DTYPE
        )
        row_starts = np.empty(num_constraints + 1, dtype=INDEX_ARRAY_DTYPE)
        col_indices = np.empty(total_nonzeros, dtype=INDEX_ARRAY_DTYPE)
        values = np.empty(total_nonzeros, dtype=CONSTRAINT_ARRAY_DTYPE)

        row_index = 0
        nonzero_index = 0

        # Build constraints for each transition and source edge
        for transition_index, transition in enumerate(self.transitions):
            from_operation_index = transition.from_operation_index

            for from_edge_index in range(self.edge_count):
                row_starts[row_index] = nonzero_index

                # Add z variable coefficients: +1 for each z_{trans, from_edge, to_edge}
                for to_edge_index in range(self.edge_count):
                    z_variable_index = self._get_z_var_index(
                        transition_index, from_edge_index, to_edge_index
                    )
                    col_indices[nonzero_index] = z_variable_index
                    values[nonzero_index] = COEFFICIENT_POSITIVE
                    nonzero_index += 1

                # Add y variable coefficient: -1 for y_{from_op, from_edge}
                y_variable_index = self._get_y_var_index(
                    from_operation_index, from_edge_index
                )
                col_indices[nonzero_index] = y_variable_index
                values[nonzero_index] = COEFFICIENT_NEGATIVE
                nonzero_index += 1

                row_index += 1

        row_starts[num_constraints] = total_nonzeros

        # Add all constraints in one batch call
        highs_model.addRows(
            num_constraints,
            lower_bounds,
            upper_bounds,
            total_nonzeros,
            row_starts,
            col_indices,
            values,
        )

    def _add_inflow_constraints(self, highs_model: highspy.Highs) -> None:
        """Add inflow flow conservation constraints.

        For each transition and each destination edge, ensures that the total
        flow arriving at that edge equals the assignment of the destination
        operation to that edge.

        This complements the outflow constraints to create a balanced flow
        structure that improves the LP relaxation quality (reduces integrality
        gap).

        Constraint form:
            Σ_{from_edge} z_{transition, from_edge, to_edge} = y_{to_op, to_edge}

        Args:
            highs_model: The HiGHS model instance to add constraints to.
        """
        # Calculate constraint dimensions (same structure as outflow)
        num_constraints = self.transition_count * self.edge_count
        nonzeros_per_constraint = self.edge_count + 1  # E z-variables + 1 y-variable
        total_nonzeros = num_constraints * nonzeros_per_constraint

        # Pre-allocate CSR format arrays
        lower_bounds = np.full(
            num_constraints, EQUALITY_BOUND_ZERO, dtype=CONSTRAINT_ARRAY_DTYPE
        )
        upper_bounds = np.full(
            num_constraints, EQUALITY_BOUND_ZERO, dtype=CONSTRAINT_ARRAY_DTYPE
        )
        row_starts = np.empty(num_constraints + 1, dtype=INDEX_ARRAY_DTYPE)
        col_indices = np.empty(total_nonzeros, dtype=INDEX_ARRAY_DTYPE)
        values = np.empty(total_nonzeros, dtype=CONSTRAINT_ARRAY_DTYPE)

        row_index = 0
        nonzero_index = 0

        # Build constraints for each transition and destination edge
        for transition_index, transition in enumerate(self.transitions):
            to_operation_index = transition.to_operation_index

            for to_edge_index in range(self.edge_count):
                row_starts[row_index] = nonzero_index

                # Add z variable coefficients: +1 for each z_{trans, from_edge, to_edge}
                # Note: we sum over from_edge (unlike outflow which sums over to_edge)
                for from_edge_index in range(self.edge_count):
                    z_variable_index = self._get_z_var_index(
                        transition_index, from_edge_index, to_edge_index
                    )
                    col_indices[nonzero_index] = z_variable_index
                    values[nonzero_index] = COEFFICIENT_POSITIVE
                    nonzero_index += 1

                # Add y variable coefficient: -1 for y_{to_op, to_edge}
                y_variable_index = self._get_y_var_index(
                    to_operation_index, to_edge_index
                )
                col_indices[nonzero_index] = y_variable_index
                values[nonzero_index] = COEFFICIENT_NEGATIVE
                nonzero_index += 1

                row_index += 1

        row_starts[num_constraints] = total_nonzeros

        # Add all constraints in one batch call
        highs_model.addRows(
            num_constraints,
            lower_bounds,
            upper_bounds,
            total_nonzeros,
            row_starts,
            col_indices,
            values,
        )

    def _add_mccormick_lower_bounds(self, highs_model: highspy.Highs) -> None:
        """Add McCormick lower bound constraints for z variables.

        For each z variable, adds the constraint:
            z_{transition, from_edge, to_edge} ≥ y_{from_op, from_edge} + y_{to_op, to_edge} - 1

        This prevents the LP from setting z=0 when both y variables are 1,
        forcing flow to match the actual operation assignments. This is the
        key constraint for closing the integrality gap.

        Rearranged form for HiGHS (lower bound on expression):
            z - y₁ - y₂ ≥ -1

        Args:
            highs_model: The HiGHS model instance to add constraints to.
        """
        # One constraint per z variable
        num_constraints = self.z_var_count
        nonzeros_per_constraint = 3  # z, y₁, y₂
        total_nonzeros = num_constraints * nonzeros_per_constraint

        # Pre-allocate CSR format arrays
        lower_bounds = np.full(
            num_constraints, COEFFICIENT_NEGATIVE, dtype=CONSTRAINT_ARRAY_DTYPE
        )  # -1
        upper_bounds = np.full(
            num_constraints, highspy.kHighsInf, dtype=CONSTRAINT_ARRAY_DTYPE
        )  # +∞
        row_starts = np.empty(num_constraints + 1, dtype=INDEX_ARRAY_DTYPE)
        col_indices = np.empty(total_nonzeros, dtype=INDEX_ARRAY_DTYPE)
        values = np.empty(total_nonzeros, dtype=CONSTRAINT_ARRAY_DTYPE)

        row_index = 0
        nonzero_index = 0

        for transition_index, transition in enumerate(self.transitions):
            from_operation_index = transition.from_operation_index
            to_operation_index = transition.to_operation_index

            for from_edge_index in range(self.edge_count):
                for to_edge_index in range(self.edge_count):
                    row_starts[row_index] = nonzero_index

                    # +1 * z_{transition, from_edge, to_edge}
                    z_variable_index = self._get_z_var_index(
                        transition_index, from_edge_index, to_edge_index
                    )
                    col_indices[nonzero_index] = z_variable_index
                    values[nonzero_index] = COEFFICIENT_POSITIVE
                    nonzero_index += 1

                    # -1 * y_{from_op, from_edge}
                    y1_variable_index = self._get_y_var_index(
                        from_operation_index, from_edge_index
                    )
                    col_indices[nonzero_index] = y1_variable_index
                    values[nonzero_index] = COEFFICIENT_NEGATIVE
                    nonzero_index += 1

                    # -1 * y_{to_op, to_edge}
                    y2_variable_index = self._get_y_var_index(
                        to_operation_index, to_edge_index
                    )
                    col_indices[nonzero_index] = y2_variable_index
                    values[nonzero_index] = COEFFICIENT_NEGATIVE
                    nonzero_index += 1

                    row_index += 1

        row_starts[num_constraints] = total_nonzeros

        # Add all constraints in one batch call
        highs_model.addRows(
            num_constraints,
            lower_bounds,
            upper_bounds,
            total_nonzeros,
            row_starts,
            col_indices,
            values,
        )

    # =========================================================================
    # Subgraph Isomorphism Inequalities (Wagner et al. 2023)
    # =========================================================================

    def _build_relaxed_graph(self, d: int) -> nx.Graph:
        """Build the d-th relaxed hardware graph H^d.

        Definition 3.9 from Wagner et al. (2023): The d-th relaxed graph H^d
        has the same vertices as H, but edges connect any pair of nodes within
        distance d+1 in the original hardware graph.

        The relaxed graph H^d captures which node pairs can communicate within
        d+1 SWAP operations. H^0 = H (original), H^1 adds edges for distance-2
        pairs, etc.

        Args:
            d: Relaxation parameter (non-negative integer).

        Returns:
            NetworkX Graph representing H^d with integer node indices and
            edges for all node pairs within distance d+1.

        Reference:
            Wagner, Bärmann, Liers, Weissenbäck (2023). "Improving Quantum
            Computation by Optimized Qubit Routing", J. Optim. Theory Appl.
            197:1161-1194, Definition 3.9.
        """
        relaxed_graph: nx.Graph[int] = nx.Graph()
        hardware_nodes = list(self.coupling_map.nodes())

        # Use integer indices for compatibility with connectivity graph
        node_indices = [node.index for node in hardware_nodes]
        relaxed_graph.add_nodes_from(node_indices)

        threshold_distance = d + 1

        for node_u in hardware_nodes:
            for node_v in hardware_nodes:
                if node_u.index >= node_v.index:
                    continue  # Avoid self-loops and duplicates

                distance = self.distance_matrix.get((node_u, node_v), float("inf"))
                if distance <= threshold_distance:
                    relaxed_graph.add_edge(node_u.index, node_v.index)

        return relaxed_graph

    def _build_connectivity_graph(self, operation_indices: list[int]) -> nx.Graph:
        """Build the connectivity graph for a subset of operations.

        Definition 3.6 from Wagner et al. (2023): For a set of gates, the
        connectivity graph C(G̃) has logical qubits as vertices and edges
        connecting qubits that participate together in a gate.

        The connectivity graph captures the structure of qubit interactions
        in the circuit subset. If this graph cannot be embedded in the
        hardware graph, routing requires movement.

        Args:
            operation_indices: Indices of operations to include.

        Returns:
            NetworkX Graph with logical qubit indices as nodes and edges
            for each operation's qubit pair.

        Reference:
            Wagner, Bärmann, Liers, Weissenbäck (2023). "Improving Quantum
            Computation by Optimized Qubit Routing", J. Optim. Theory Appl.
            197:1161-1194, Definition 3.6.
        """
        connectivity_graph: nx.Graph[int] = nx.Graph()

        for operation_index in operation_indices:
            operation = self.operations[operation_index]
            qubits = operation.qubits_participating

            if len(qubits) >= 2:
                qubit_a = qubits[0].index
                qubit_b = qubits[1].index
                connectivity_graph.add_edge(qubit_a, qubit_b)

        return connectivity_graph

    def _check_subset_for_cut(
        self,
        operation_indices: list[int],
        relaxed_graphs: dict[int, nx.Graph],
        max_d: int,
    ) -> tuple[list[int], int] | None:
        """Check if an operation subset yields a violated SGI inequality.

        For the given operation subset, builds the connectivity graph and
        checks for the smallest d where it cannot be embedded in H^d.

        Args:
            operation_indices: Indices of operations in the subset.
            relaxed_graphs: Pre-computed relaxed graphs indexed by d.
            max_d: Maximum d value to check.

        Returns:
            Tuple (operation_indices, min_distance) if a cut is found,
            None otherwise.
        """
        connectivity_graph = self._build_connectivity_graph(operation_indices)

        if connectivity_graph.number_of_edges() == 0:
            return None

        # Find minimum d where embedding fails
        for d in range(max_d + 1):
            relaxed = relaxed_graphs[d]

            # Use VF2 algorithm for subgraph monomorphism check.
            # NOTE: subgraph_is_isomorphic() checks INDUCED subgraph isomorphism,
            # but we need general subgraph (monomorphism). Use iterator instead.
            # Reference: https://networkx.org/documentation/stable/reference/algorithms/isomorphism.vf2.html
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                relaxed, connectivity_graph
            )

            # Check if any subgraph monomorphism exists
            has_embedding = any(True for _ in matcher.subgraph_monomorphisms_iter())

            if not has_embedding:
                # Found a violated inequality: movement must be at least d+1
                min_distance = d + 1
                return (operation_indices, min_distance)

        return None

    def _find_sgi_cuts(
        self,
        max_subset_size: int = 8,
        max_diameter: int | None = None,
    ) -> list[tuple[list[int], int]]:
        """Find violated subgraph isomorphism inequalities.

        Lemma 3.10 from Wagner et al. (2023): If the connectivity graph C
        of an operation subset is NOT subgraph-isomorphic to the d-th relaxed
        graph H^d, then the operations must involve total movement of at
        least d+1.

        This method searches for operation subsets whose connectivity graphs
        cannot be embedded in H^d for various values of d. Each violated
        inequality provides a lower bound on the objective.

        The search strategy examines:
        1. Single layers (gates that must execute together)
        2. Consecutive layer pairs/triples
        3. Operation chains connected by qubit sharing

        Args:
            max_subset_size: Maximum number of operations to consider in a
                subset. Larger values find more cuts but increase runtime.
            max_diameter: Maximum d value to check. If None, uses the graph
                diameter.

        Returns:
            List of (operation_indices, min_distance) tuples, where each
            represents a valid inequality: total movement ≥ min_distance.

        Reference:
            Wagner, Bärmann, Liers, Weissenbäck (2023). "Improving Quantum
            Computation by Optimized Qubit Routing", J. Optim. Theory Appl.
            197:1161-1194, Lemma 3.10.
        """
        cuts: list[tuple[list[int], int]] = []

        if max_diameter is None:
            max_diameter = int(
                max(
                    dist
                    for dist in self.distance_matrix.values()
                    if dist != float("inf")
                )
            )

        # Pre-compute relaxed graphs for each d value
        relaxed_graphs = {
            d: self._build_relaxed_graph(d) for d in range(max_diameter + 1)
        }

        # Strategy 1: Check individual layers
        for layer_operations in self.operation_layers:
            if len(layer_operations) > max_subset_size:
                continue

            cut = self._check_subset_for_cut(
                layer_operations, relaxed_graphs, max_diameter
            )
            if cut is not None:
                cuts.append(cut)

        # Strategy 2: Check consecutive layer pairs/triples
        for layer_start in range(len(self.operation_layers)):
            for layer_end in range(layer_start + 1, len(self.operation_layers)):
                operation_subset = []
                for layer_idx in range(layer_start, layer_end + 1):
                    operation_subset.extend(self.operation_layers[layer_idx])

                if len(operation_subset) > max_subset_size:
                    break  # No point checking larger ranges

                cut = self._check_subset_for_cut(
                    operation_subset, relaxed_graphs, max_diameter
                )
                if cut is not None:
                    cuts.append(cut)

        # Strategy 3: Check connected operation chains (via transitions)
        for transition in self.transitions:
            chain = [transition.from_operation_index, transition.to_operation_index]
            cut = self._check_subset_for_cut(chain, relaxed_graphs, max_diameter)
            if cut is not None:
                cuts.append(cut)

        _logger.info(f"Found {len(cuts)} subgraph isomorphism cuts")
        return cuts

    def _add_sgi_constraints(self, highs_model: highspy.Highs) -> None:
        """Add subgraph isomorphism inequality constraints.

        For each violated SGI cut, adds a constraint requiring the total
        movement cost to be at least the minimum distance bound.

        Constraint form (Inequality 12 from Wagner):
            Σ cost(e₁, e₂) · z_{trans, e₁, e₂} ≥ min_distance

        where the sum is over all transitions involving operations in the
        subset.

        Args:
            highs_model: The HiGHS model instance to add constraints to.
        """
        cuts = self._find_sgi_cuts()

        if not cuts:
            _logger.info("No SGI cuts found to add")
            return

        # Build operation → transitions mapping for efficient lookup
        operation_to_transitions: dict[int, list[int]] = defaultdict(list)
        for trans_idx, transition in enumerate(self.transitions):
            operation_to_transitions[transition.from_operation_index].append(trans_idx)
            operation_to_transitions[transition.to_operation_index].append(trans_idx)

        constraints_added = 0

        for operation_indices, min_distance in cuts:
            # Collect all transitions involving these operations
            relevant_transition_indices = set()
            for operation_index in operation_indices:
                relevant_transition_indices.update(
                    operation_to_transitions[operation_index]
                )

            if not relevant_transition_indices:
                continue

            # Build constraint: Σ cost · z ≥ min_distance
            row_indices: list[int] = []
            row_values: list[float] = []

            for trans_idx in relevant_transition_indices:
                transition = self.transitions[trans_idx]
                for from_edge_idx in range(self.edge_count):
                    from_edge = self.edges[from_edge_idx]
                    for to_edge_idx in range(self.edge_count):
                        to_edge = self.edges[to_edge_idx]

                        # Compute cost for this z variable
                        cost = 0.0
                        for qubit in transition.shared_qubits:
                            from_pos = self._get_qubit_position_in_edge(
                                qubit, transition.from_operation_index, from_edge
                            )
                            to_pos = self._get_qubit_position_in_edge(
                                qubit, transition.to_operation_index, to_edge
                            )
                            cost += self.distance_matrix[(from_pos, to_pos)]

                        if cost > 0:
                            z_idx = self._get_z_var_index(
                                trans_idx, from_edge_idx, to_edge_idx
                            )
                            row_indices.append(z_idx)
                            row_values.append(cost)

            if row_indices:
                highs_model.addRow(
                    float(min_distance),  # lower bound
                    highspy.kHighsInf,  # upper bound (no limit)
                    len(row_indices),
                    row_indices,
                    row_values,
                )
                constraints_added += 1

        _logger.info(f"Added {constraints_added} SGI constraints")

    def _set_objective(self, highs_model: highspy.Highs) -> None:
        """Set the objective function: minimise summed distances.

        For each z variable, the cost is the sum of shortest-path distances
        for ALL shared qubits in the transition. This correctly penalises
        total qubit movement when operations share multiple qubits.

        Args:
            highs_model: The HiGHS model instance to set the objective on.
        """
        highs_model.changeObjectiveSense(highspy.ObjSense.kMinimize)

        # Pre-allocate arrays for batch objective setting
        z_variable_indices = np.empty(self.z_var_count, dtype=INDEX_ARRAY_DTYPE)
        z_variable_costs = np.empty(self.z_var_count, dtype=CONSTRAINT_ARRAY_DTYPE)

        variable_counter = 0

        for transition_index, transition in enumerate(self.transitions):
            for from_edge_index in range(self.edge_count):
                from_edge = self.edges[from_edge_index]

                for to_edge_index in range(self.edge_count):
                    to_edge = self.edges[to_edge_index]

                    # Sum distances for ALL shared qubits in this transition
                    total_distance = 0.0
                    for qubit in transition.shared_qubits:
                        from_position = self._get_qubit_position_in_edge(
                            qubit, transition.from_operation_index, from_edge
                        )
                        to_position = self._get_qubit_position_in_edge(
                            qubit, transition.to_operation_index, to_edge
                        )
                        total_distance += self.distance_matrix[
                            (from_position, to_position)
                        ]

                    # Record variable index and cost
                    z_variable_index = self._get_z_var_index(
                        transition_index, from_edge_index, to_edge_index
                    )
                    z_variable_indices[variable_counter] = z_variable_index
                    z_variable_costs[variable_counter] = total_distance
                    variable_counter += 1

        # Set all z variable costs in one batch call
        highs_model.changeColsCost(
            self.z_var_count, z_variable_indices, z_variable_costs
        )

    def _add_variables(self, highs_model: highspy.Highs) -> None:
        """Add decision variables to the HiGHS model.

        Adds y variables (operation-edge assignment) with [0,1] bounds and
        z variables (flow) with [0, infinity) bounds.

        Args:
            highs_model: The HiGHS model instance to add variables to.
        """
        # Add y variables with [0, 1] bounds
        y_lower_bounds = np.full(
            self.y_var_count, Y_VARIABLE_LOWER_BOUND, dtype=np.float64
        )
        y_upper_bounds = np.full(
            self.y_var_count, Y_VARIABLE_UPPER_BOUND, dtype=np.float64
        )
        highs_model.addVars(self.y_var_count, y_lower_bounds, y_upper_bounds)

        # Add z variables with [0, infinity) bounds
        z_lower_bounds = np.full(
            self.z_var_count, Z_VARIABLE_LOWER_BOUND, dtype=np.float64
        )
        z_upper_bounds = np.full(self.z_var_count, highspy.kHighsInf, dtype=np.float64)
        highs_model.addVars(self.z_var_count, z_lower_bounds, z_upper_bounds)

    def _find_best_edge_pair_for_transition(
        self,
        transition_index: int,
        column_values: list[float],
    ) -> tuple[int, int]:
        """Find the edge pair with maximum z value for a transition.

        Uses argmax to select the (from_edge, to_edge) pair with highest
        z variable value, which is robust to fractional LP solutions.

        Args:
            transition_index: Index of the transition in self.transitions.
            column_values: Solution variable values from HiGHS.

        Returns:
            Tuple of (from_edge_index, to_edge_index) for the best pair.
        """
        z_block_start = self._get_z_var_index(transition_index, 0, 0)
        z_block_size = self.edge_count * self.edge_count

        z_block = np.array(column_values[z_block_start : z_block_start + z_block_size])
        best_flat_index = np.argmax(z_block)

        from_edge_index, to_edge_index = divmod(best_flat_index, self.edge_count)
        return int(from_edge_index), int(to_edge_index)

    def _extract_swaps(
        self,
        column_values: list[float],
        operation_edge_assignment: dict[int, PhysicalEdge],
    ) -> dict[int, list[quariadne.circuit.PhysicalSwap]]:
        """Extract SWAP sequences from z flow variables.

        For each transition, finds the best (from_edge, to_edge) pair and
        generates SWAPs for ALL shared qubits along their shortest paths.

        Args:
            column_values: Solution variable values from HiGHS.
            operation_edge_assignment: Mapping from operation index to edge.

        Returns:
            Dictionary mapping operation index to list of PhysicalSwap objects.
            Swaps for operation i should be inserted before that operation.
        """
        inserted_swaps: dict[int, list[quariadne.circuit.PhysicalSwap]] = defaultdict(
            list
        )

        # Get cached shortest paths
        coupling_edges = frozenset(
            (edge[0].index, edge[1].index) for edge in self.coupling_map.edges()
        )
        cached_paths = compute_all_pairs_shortest_paths(coupling_edges)

        for transition_index, transition in enumerate(self.transitions):
            target_operation = transition.to_operation_index

            # Use actual Y assignments (not Z-derived edges which may be inconsistent
            # after LP relaxation and rounding)
            from_edge = operation_edge_assignment[transition.from_operation_index]
            to_edge = operation_edge_assignment[transition.to_operation_index]

            # Generate SWAPs for EACH shared qubit
            for qubit in transition.shared_qubits:
                from_position = self._get_qubit_position_in_edge(
                    qubit, transition.from_operation_index, from_edge
                )
                to_position = self._get_qubit_position_in_edge(
                    qubit, transition.to_operation_index, to_edge
                )

                # If positions differ, generate swap path
                if from_position != to_position:
                    path_key = (from_position.index, to_position.index)
                    swap_path = cached_paths[path_key]

                    # Create SWAP for each step in the path (skip duplicates)
                    for step_index in range(len(swap_path) - 1):
                        current_position = swap_path[step_index]
                        next_position = swap_path[step_index + 1]
                        swap = quariadne.circuit.PhysicalSwap(
                            quariadne.circuit.PhysicalQubit(current_position),
                            quariadne.circuit.PhysicalQubit(next_position),
                        )
                        if swap not in inserted_swaps[target_operation]:
                            inserted_swaps[target_operation].append(swap)

        return inserted_swaps

    def _build_model(self) -> highspy.Highs:
        """Build the complete HiGHS LP model.

        Configures solver options, adds variables, constraints, and objective.

        Returns:
            Configured HiGHS model ready for solving.
        """
        highs_model = highspy.Highs()

        # Configure solver options
        highs_model.setOptionValue("output_flag", self.solver_options.output_flag)
        highs_model.setOptionValue("log_to_console", self.solver_options.log_to_console)
        highs_model.setOptionValue("solver", self.solver_options.solver)
        highs_model.setOptionValue(
            "hipo_system_solver", self.solver_options.hipo_system_solver
        )
        highs_model.setOptionValue("threads", self.solver_options.threads)

        # Add variables and constraints
        self._add_variables(highs_model)
        self._add_operation_uniqueness_constraints(highs_model)
        self._add_position_exclusivity_constraints(highs_model)
        self._add_flow_conservation_constraints(highs_model)  # outflow
        self._add_inflow_constraints(highs_model)  # inflow (for tighter LP relaxation)
        self._add_mccormick_lower_bounds(highs_model)  # z ≥ y₁ + y₂ - 1
        self._add_sgi_constraints(
            highs_model
        )  # subgraph isomorphism cuts (Wagner 2023)
        self._add_fixed_operation_constraints(highs_model)
        self._set_objective(highs_model)

        return highs_model

    def run(self) -> UnifiedLPResult:
        """Solve the LP and return the routing result.

        Builds the complete LP model, solves it using HiGHS, and extracts
        the solution including operation-to-edge assignments, initial/final
        qubit mappings, and SWAP sequences.

        Returns:
            UnifiedLPResult containing all routing information.

        Raises:
            RuntimeError: If the LP optimisation fails to find an optimal solution.
        """
        start_time = time.perf_counter()

        # Build and solve the LP model
        highs_model = self._build_model()
        highs_model.run()

        # Verify solution status
        model_status = highs_model.getModelStatus()
        if model_status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"LP optimisation failed with status: {model_status}")

        # Extract solution data
        solution = highs_model.getSolution()
        column_values = solution.col_value
        objective_value = highs_model.getInfo().objective_function_value

        # Extract routing results
        operation_edge_assignment = self._extract_operation_edges(column_values)
        inserted_swaps = self._extract_swaps(column_values, operation_edge_assignment)
        initial_mapping = self._extract_initial_mapping(operation_edge_assignment)
        final_mapping = self._extract_final_mapping(operation_edge_assignment)
        edge_by_qubits = self._build_operation_to_edge_by_qubits(
            operation_edge_assignment
        )

        elapsed_time = time.perf_counter() - start_time
        _logger.info(f"Unified LP router completed in {elapsed_time:.3f}s")
        _logger.info(f"Objective value: {objective_value}")

        return UnifiedLPResult(
            objective_value=objective_value,
            operation_edge_assignment=operation_edge_assignment,
            operation_to_edge_by_qubits=edge_by_qubits,
            initial_mapping=initial_mapping,
            final_mapping=final_mapping,
            operations=self.operations,
            inserted_swaps=inserted_swaps,
        )

    def run_with_integer_rounding(self) -> UnifiedLPResult:
        """Solve LP and round all y-variables to integer assignments.

        This method provides a simple rounding approach:
        1. Solve the full LP relaxation (with SGI constraints)
        2. For each operation, pick the edge with maximum y value (argmax)
        3. Create a final LP with all assignments fixed to verify feasibility
        4. Extract SWAPs from the fixed solution

        The SGI constraints help the LP find better fractional solutions,
        which should round to better integer solutions.

        Returns:
            UnifiedLPResult with integer edge assignments.
        """
        start_time = time.perf_counter()

        # Step 1: Solve the LP relaxation
        _logger.info("Solving LP relaxation with SGI constraints")
        highs_model = self._build_model()
        highs_model.run()

        model_status = highs_model.getModelStatus()
        if model_status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"LP optimisation failed: {model_status}")

        solution = highs_model.getSolution()
        column_values = solution.col_value
        lp_objective = highs_model.getInfo().objective_function_value
        _logger.info(f"LP objective (fractional): {lp_objective:.4f}")

        # Step 2: Round all y-variables to integer (argmax per operation)
        all_fixed_edges: dict[int, int] = {}
        for op_idx in range(self.operation_count):
            best_edge_idx = self._find_best_edge_for_operation(op_idx, column_values)
            all_fixed_edges[op_idx] = best_edge_idx

        # Step 3: Extract solution directly using rounded assignments
        # Map operation_idx -> PhysicalEdge
        operation_edge_assignment: dict[
            int, tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit]
        ] = {}
        for op_idx, edge_idx in all_fixed_edges.items():
            operation_edge_assignment[op_idx] = self.edges[edge_idx]

        # Extract SWAPs using the rounded assignments
        inserted_swaps = self._extract_swaps(column_values, operation_edge_assignment)

        # Extract mappings
        initial_mapping = self._extract_initial_mapping(operation_edge_assignment)
        final_mapping = self._extract_final_mapping(operation_edge_assignment)
        edge_by_qubits = self._build_operation_to_edge_by_qubits(
            operation_edge_assignment
        )

        # Count SWAPs for logging
        total_swaps = sum(len(swaps) for swaps in inserted_swaps.values())
        elapsed_time = time.perf_counter() - start_time
        _logger.info(f"Integer rounding completed in {elapsed_time:.3f}s")
        _logger.info(f"Total SWAPs: {total_swaps}")

        return UnifiedLPResult(
            objective_value=lp_objective,
            operation_edge_assignment=operation_edge_assignment,
            operation_to_edge_by_qubits=edge_by_qubits,
            initial_mapping=initial_mapping,
            final_mapping=final_mapping,
            operations=self.operations,
            inserted_swaps=inserted_swaps,
        )

    def _find_best_edge_for_operation(
        self,
        operation_idx: int,
        column_values: list[float],
    ) -> int:
        """Find the edge with maximum y value for an operation.

        Uses argmax rounding to select the best edge assignment for a
        single operation from the LP solution.

        Args:
            operation_idx: Index of the operation.
            column_values: Solution variable values from HiGHS.

        Returns:
            Index of the edge with highest y value.
        """
        best_edge_idx = 0
        best_y_value = -1.0

        for edge_idx in range(self.edge_count):
            y_idx = self._get_y_var_index(operation_idx, edge_idx)
            y_value = column_values[y_idx]
            if y_value > best_y_value:
                best_y_value = y_value
                best_edge_idx = edge_idx

        return best_edge_idx
