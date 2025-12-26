"""Bipartite Allocation LP Router for quantum circuit routing.

This module implements a Linear Programming formulation for the token allocation
problem in quantum circuit routing using bipartite matching concepts. The approach
treats qubit routing as a network flow problem where operations are assigned to
hardware edges and qubit movements are tracked through flow conservation.

Mathematical Formulation
------------------------
The LP relaxation (with binary y variables) solves the following problem:

**Decision Variables:**
    - ``y_{o,e} ∈ {0,1}``: Binary variable indicating operation o is assigned to
      directed edge e = (p1, p2) on the coupling graph.
    - ``z^l_{o,e',e} ≥ 0``: Continuous flow variable linking edge e' at operation o'
      to edge e at operation o for logical qubit l. Represents qubit movement.

**Constraints:**
    1. Operation Uniqueness: ``Σ_e y_{o,e} = 1`` for each operation o.
       Each operation must be assigned to exactly one edge.

    2. Flow Conservation (Outflow): ``Σ_e z^l_{o,e',e} = y_{o',e'}`` for each e'.
       Total flow leaving edge e' must equal the assignment to that edge.

    3. Flow Conservation (Inflow): ``Σ_{e'} z^l_{o,e',e} = y_{o,e}`` for each e.
       Total flow entering edge e must equal the assignment to that edge.

**Objective Function:**
    Minimise ``Σ dist(pos_l(e'), pos_l(e)) · z^l_{o,e',e}``

    where ``pos_l(e)`` is the physical position where logical qubit l resides
    when assigned to edge e. The distance function returns the shortest path
    length on the undirected coupling graph (number of SWAPs needed).

LP Relaxation Properties
------------------------
The constraint matrix exhibits total unimodularity for the flow conservation
substructure. When the y variables are fixed, the z subproblem becomes a
transportation problem with integral extreme points. The HiGHS solver typically
finds integral solutions even without explicit integrality constraints on z.

Known Limitation
----------------
**BUG**: The current formulation lacks global position consistency constraints.
Each operation is assigned to an edge independently, without enforcing that:

1. Different logical qubits occupy different physical positions at any time
2. Qubit positions are globally consistent across all operations

This allows the LP to "reuse" physical positions for different logical qubits
at different operations, leading to infeasible solutions. For example, if q2 is
at position 4 in operation 1 and q4 is at position 4 in operation 3, the LP
treats these as valid even though physically impossible.

The fix requires adding position exclusivity constraints that track qubit
positions across all operations, not just between consecutive operations
involving the same qubit. This is a TODO for future work.

Implementation Notes
--------------------
- Uses HiGHS solver (highspy) for efficient LP/MIP optimisation.
- Distance matrix is cached at module level since hardware topology is stable.
- The BipartiteRouting pass in qiskit_passes.py uses ``operation_to_edge_by_qubits``
  to look up edge assignments, avoiding ordering mismatches between LP and DAG.
- QUEKO benchmarks (designed for optimal routing) typically yield zero objective
  value, confirming the LP finds swap-free assignments when they exist.

References
----------
- HiGHS Python Interface: https://ergo-code.github.io/HiGHS/dev/interfaces/python/
- Qiskit Transpiler: https://docs.quantum.ibm.com/api/qiskit/transpiler

Example
-------
>>> from quariadne.bipartite_allocation_lp import BipartiteAllocationRouter
>>> router = BipartiteAllocationRouter(coupling_graph, quantum_circuit)
>>> result = router.run()
>>> print(f"Objective: {result.objective_value}")
>>> print(f"Initial mapping: {result.initial_mapping}")
"""

from __future__ import annotations

import copy
import functools
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import highspy
import networkx as nx

import quariadne.circuit

if TYPE_CHECKING:
    import qiskit.transpiler


@functools.lru_cache(maxsize=8)
def _compute_all_pairs_distances(
    edges: frozenset[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Compute shortest path distances for a coupling graph.

    Uses LRU cache to avoid recomputation for the same hardware topology.
    The cache key is a frozenset of edge tuples identifying the graph structure.

    Args:
        edges: Frozenset of (from_index, to_index) tuples defining the coupling graph.

    Returns:
        Dictionary mapping (source_index, target_index) to shortest path length.

    See Also:
        - https://docs.python.org/3.13/library/functools.html#functools.lru_cache
        - https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html
    """
    # Reconstruct graph from edges
    graph: nx.Graph[int] = nx.Graph()
    for from_idx, to_idx in edges:
        graph.add_edge(from_idx, to_idx)

    # Compute all-pairs shortest paths
    distances: dict[tuple[int, int], int] = {}
    all_pairs = dict(nx.all_pairs_shortest_path_length(graph))

    for source, targets in all_pairs.items():
        for target, length in targets.items():
            distances[(source, target)] = length

    return distances


class BipartiteVariableType(Enum):
    """Enum for variable types in the bipartite allocation LP."""

    OPERATION_EDGE = "operation_edge"  # y_{o,e}
    FLOW = "flow"  # z^l_{o,e',e}


@dataclass
class QubitOperationInfo:
    """Tracks which operations involve each qubit and their ordering.

    Attributes:
        qubit: The logical qubit being tracked
        operation_indices: List of operation indices involving this qubit (in order)
    """

    qubit: quariadne.circuit.LogicalQubit
    operation_indices: list[int] = field(default_factory=list)


@dataclass
class FlowTransition:
    """Represents a flow transition for a qubit between consecutive operations.

    Attributes:
        qubit: The logical qubit transitioning
        from_operation_index: The previous operation index involving this qubit
        to_operation_index: The current operation index involving this qubit
    """

    qubit: quariadne.circuit.LogicalQubit
    from_operation_index: int
    to_operation_index: int


@dataclass
class BipartiteAllocationResult:
    """Result container for bipartite allocation LP optimisation.

    Contains the solution mapping operations to edges, the derived
    initial qubit mapping, and the swap sequence extracted from z flow variables.

    Attributes:
        objective_value: The optimal objective value (total distance)
        operation_edge_assignment: Dict mapping operation index to assigned edge
        operation_to_edge_by_qubits: Dict mapping (qubit1, qubit2, occurrence) to edge
        initial_mapping: Initial logical-to-physical qubit mapping
        operations: List of operations in LP order for reference
        inserted_swaps: Dict mapping operation index to list of PhysicalSwap objects
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
    operations: list[quariadne.circuit.QuantumOperation]
    inserted_swaps: dict[int, list[quariadne.circuit.PhysicalSwap]]


def get_coupling_graph(coupling_map: qiskit.transpiler.CouplingMap) -> nx.DiGraph:
    """Convert IBM coupling map to NetworkX DiGraph coupling map.

    Converts a Qiskit CouplingMap object to a NetworkX DiGraph using
    Quariadne's PhysicalQubit representation for nodes and edges.

    Args:
        coupling_map: Qiskit CouplingMap object representing physical qubit
                     connectivity (e.g., backend.coupling_map)

    Returns:
        NetworkX DiGraph with PhysicalQubit nodes and edges representing
        physical qubit connectivity on the hardware backend.
    """
    coupling_graph_nx: nx.DiGraph[quariadne.circuit.PhysicalQubit] = nx.DiGraph()

    coupling_map_qubits = coupling_map.physical_qubits
    physical_qubits = tuple(
        quariadne.circuit.PhysicalQubit(coupling_map_qubit)
        for coupling_map_qubit in coupling_map_qubits
    )

    coupling_map_edgelist = coupling_map.graph.edge_list()
    physical_qubits_connections = tuple(
        (
            quariadne.circuit.PhysicalQubit(coupling_map_from),
            quariadne.circuit.PhysicalQubit(coupling_map_to),
        )
        for coupling_map_from, coupling_map_to in coupling_map_edgelist
    )

    coupling_graph_nx.add_nodes_from(physical_qubits)
    coupling_graph_nx.add_edges_from(physical_qubits_connections)

    return coupling_graph_nx


class BipartiteAllocationRouter:
    """Linear Programming router using bipartite matching formulation.

    Implements an LP-based approach where operations are matched to coupling edges
    with flow conservation constraints tracking qubit positions.

    Mathematical Formulation:
        Variables:
            y_{o,e} ∈ {0,1}: operation o assigned to directed edge e
            z^l_{o,e',e} ≥ 0: flow from edge e' to edge e for qubit l at operation o

        Constraints:
            Σ_e y_{o,e} = 1              (each operation on exactly one edge)
            Σ_o y_{o,e} ≤ 1 per layer    (each edge used at most once per layer)
            Flow conservation on z linking consecutive y's

        Objective:
            Minimise Σ dist(pos_l(e'), pos_l(e)) · z^l_{o,e',e}
            where pos_l(e) is the endpoint of edge e where qubit l sits

    Attributes:
        coupling_map: NetworkX DiGraph representing hardware connectivity
        routed_circuit: Abstract quantum circuit to be routed
        operations: List of two-qubit operations to route
        edges: List of directed edges in the coupling map
        distance_matrix: Shortest path distances between physical qubits
    """

    def __init__(
        self,
        coupling_map: nx.DiGraph,
        quantum_circuit: quariadne.circuit.AbstractQuantumCircuit,
    ) -> None:
        """Initialise the bipartite allocation router.

        Args:
            coupling_map: NetworkX DiGraph representing physical qubit connectivity
            quantum_circuit: Abstract quantum circuit to be routed
        """
        self.coupling_map = coupling_map
        self.qubit_count = coupling_map.number_of_nodes()
        self.routed_circuit = copy.deepcopy(quantum_circuit)

        self._add_dummy_qubits()

        self.operations = self._get_two_qubit_operations()
        self.operation_count = len(self.operations)

        self.edges = list(self.coupling_map.edges())
        self.edge_count = len(self.edges)

        # Build edge index mapping for quick lookup
        self.edge_to_index: dict[
            tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit], int
        ] = {edge: idx for idx, edge in enumerate(self.edges)}

        # Compute shortest path distances for objective function
        self.distance_matrix = self._compute_distance_matrix()

        # Build qubit operation tracking
        self.qubit_operations = self._build_qubit_operation_map()

        # Build flow transitions (pairs of consecutive operations for each qubit)
        self.flow_transitions = self._build_flow_transitions()

        # Variable indexing setup
        self._setup_variable_indexing()

    def _add_dummy_qubits(self) -> None:
        """Add dummy logical qubits to match hardware qubit count."""
        routed_circuit_qubit_count = len(self.routed_circuit.qubits)
        if routed_circuit_qubit_count < self.coupling_map.number_of_nodes():
            dummy_logical_qubits = tuple(
                quariadne.circuit.LogicalQubit(dummy_index)
                for dummy_index in range(routed_circuit_qubit_count, self.qubit_count)
            )
            self.routed_circuit.qubits = (
                self.routed_circuit.qubits + dummy_logical_qubits
            )
        elif routed_circuit_qubit_count > self.coupling_map.number_of_nodes():
            raise TypeError("Circuit has more qubits than available on hardware!")

    # Operations that are markers/directives and should be ignored for routing
    NON_GATE_OPERATIONS = frozenset({"barrier", "measure", "reset", "delay"})

    def _get_two_qubit_operations(
        self,
    ) -> list[quariadne.circuit.QuantumOperation]:
        """Extract two-qubit operations from the quantum circuit.

        Filters out single-qubit gates, barrier/measure/reset directives,
        and returns only actual two-qubit gate operations that require routing.
        """
        two_qubit_operations = []
        for operation in self.routed_circuit.operations:
            # Skip non-gate operations (barriers, measurements, etc.)
            if operation.name in self.NON_GATE_OPERATIONS:
                continue
            if len(operation.qubits_participating) == 2:
                two_qubit_operations.append(operation)
            elif len(operation.qubits_participating) > 2:
                raise TypeError(
                    f"Operations with more than 2 qubits not supported: {operation.name}"
                )
        return two_qubit_operations

    def _compute_distance_matrix(
        self,
    ) -> dict[
        tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit], float
    ]:
        """Compute shortest path distances between all pairs of physical qubits.

        Delegates to the cached `_compute_all_pairs_distances` function for the
        actual computation, then wraps the result with PhysicalQubit keys.

        Returns:
            Dictionary mapping (source, target) PhysicalQubit pairs to distance.
        """
        # Create hashable edge set for cache lookup
        edges = frozenset(
            (edge[0].index, edge[1].index) for edge in self.coupling_map.edges()
        )

        # Get cached integer-indexed distances
        index_distances = _compute_all_pairs_distances(edges)

        # Build qubit-indexed distance matrix
        distance_dict: dict[
            tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit],
            float,
        ] = {}

        for source_qubit in self.coupling_map.nodes:
            for target_qubit in self.coupling_map.nodes:
                key = (source_qubit.index, target_qubit.index)
                if key in index_distances:
                    distance_dict[(source_qubit, target_qubit)] = float(
                        index_distances[key]
                    )
                else:
                    distance_dict[(source_qubit, target_qubit)] = float("inf")

        return distance_dict

    def _build_qubit_operation_map(
        self,
    ) -> dict[quariadne.circuit.LogicalQubit, QubitOperationInfo]:
        """Build mapping from logical qubits to their participating operations.

        Returns:
            Dictionary mapping each logical qubit to its operation tracking info
        """
        qubit_ops: dict[quariadne.circuit.LogicalQubit, QubitOperationInfo] = {}

        for qubit in self.routed_circuit.qubits:
            qubit_ops[qubit] = QubitOperationInfo(qubit=qubit)

        for op_idx, operation in enumerate(self.operations):
            for qubit in operation.qubits_participating:
                qubit_ops[qubit].operation_indices.append(op_idx)

        return qubit_ops

    def _build_flow_transitions(self) -> list[FlowTransition]:
        """Build list of flow transitions for the LP formulation.

        For each qubit, creates transitions between consecutive operations
        involving that qubit.

        Returns:
            List of FlowTransition objects representing all z variable indices
        """
        transitions = []

        for qubit, op_info in self.qubit_operations.items():
            op_indices = op_info.operation_indices
            # Create transitions between consecutive operations
            for i in range(1, len(op_indices)):
                from_op = op_indices[i - 1]
                to_op = op_indices[i]
                transitions.append(
                    FlowTransition(
                        qubit=qubit,
                        from_operation_index=from_op,
                        to_operation_index=to_op,
                    )
                )

        return transitions

    def _setup_variable_indexing(self) -> None:
        """Set up variable indexing for the LP formulation.

        Variables are laid out as:
            [y_{0,0}, y_{0,1}, ..., y_{O-1,E-1}, z^{l1}_{...}, z^{l2}_{...}, ...]

        where O is operation count and E is edge count.
        """
        # y variables: operation_count × edge_count
        self.y_var_count = self.operation_count * self.edge_count
        self.y_var_offset = 0

        # z variables: for each transition, edge_count × edge_count possibilities
        # But we only need z variables for valid (e', e) pairs
        self.z_var_count = (
            len(self.flow_transitions) * self.edge_count * self.edge_count
        )
        self.z_var_offset = self.y_var_count

        self.total_var_count = self.y_var_count + self.z_var_count

        # Build transition index mapping
        self.transition_to_index: dict[tuple[int, int, int], int] = {}
        for idx, trans in enumerate(self.flow_transitions):
            key = (
                trans.qubit.index,
                trans.from_operation_index,
                trans.to_operation_index,
            )
            self.transition_to_index[key] = idx

    def _get_y_var_index(self, operation_index: int, edge_index: int) -> int:
        """Get the variable index for y_{operation, edge}.

        Args:
            operation_index: Index of the operation
            edge_index: Index of the edge

        Returns:
            Variable index in the LP formulation
        """
        return self.y_var_offset + operation_index * self.edge_count + edge_index

    def _get_z_var_index(
        self, transition_index: int, from_edge_index: int, to_edge_index: int
    ) -> int:
        """Get the variable index for z^l_{o, e', e}.

        Args:
            transition_index: Index of the flow transition
            from_edge_index: Index of the source edge (e')
            to_edge_index: Index of the target edge (e)

        Returns:
            Variable index in the LP formulation
        """
        z_local_idx = (
            transition_index * self.edge_count * self.edge_count
            + from_edge_index * self.edge_count
            + to_edge_index
        )
        return self.z_var_offset + z_local_idx

    def _get_qubit_position_in_edge(
        self,
        qubit: quariadne.circuit.LogicalQubit,
        operation_index: int,
        edge: tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit],
    ) -> quariadne.circuit.PhysicalQubit:
        """Get the physical position of a qubit when operation uses given edge.

        For a two-qubit operation on logical qubits (l1, l2) assigned to directed
        edge (p1, p2): l1 sits at p1, l2 sits at p2.

        Args:
            qubit: The logical qubit to locate
            operation_index: Index of the operation
            edge: The directed edge (source, target)

        Returns:
            Physical qubit position where the logical qubit sits
        """
        operation = self.operations[operation_index]
        left_qubit, right_qubit = operation.qubits_participating
        left_physical, right_physical = edge

        if qubit == left_qubit:
            return left_physical
        elif qubit == right_qubit:
            return right_physical
        else:
            raise ValueError(f"Qubit {qubit} not in operation {operation_index}")

    def _build_model(self) -> highspy.Highs:
        """Build the HiGHS LP model with all variables and constraints.

        Returns:
            Configured HiGHS model ready for optimisation
        """
        h = highspy.Highs()
        h.silent()

        # Add all variables
        self._add_variables(h)

        # Add constraints
        self._add_operation_uniqueness_constraints(h)
        self._add_edge_exclusivity_constraints(h)
        self._add_flow_conservation_constraints(h)

        # Set objective
        self._set_objective(h)

        return h

    def _add_variables(self, h: highspy.Highs) -> None:
        """Add all decision variables to the model.

        Args:
            h: HiGHS model instance
        """
        inf = highspy.kHighsInf

        # Add y variables (binary: operation-edge assignment)
        for _ in range(self.y_var_count):
            h.addVar(0.0, 1.0)

        # Add z variables (continuous non-negative: flow variables)
        for _ in range(self.z_var_count):
            h.addVar(0.0, inf)

        # Set y variables as binary (integers with bounds [0,1])
        for var_idx in range(self.y_var_count):
            h.changeColIntegrality(var_idx, highspy.HighsVarType.kInteger)

    def _add_operation_uniqueness_constraints(self, h: highspy.Highs) -> None:
        """Add constraints: each operation assigned to exactly one edge.

        Constraint: Σ_e y_{o,e} = 1 for each operation o

        Args:
            h: HiGHS model instance
        """
        for op_idx in range(self.operation_count):
            indices = []
            values = []
            for edge_idx in range(self.edge_count):
                var_idx = self._get_y_var_index(op_idx, edge_idx)
                indices.append(var_idx)
                values.append(1.0)

            h.addRow(1.0, 1.0, len(indices), indices, values)

    def _add_edge_exclusivity_constraints(self, h: highspy.Highs) -> None:
        """Add constraints: each edge used at most once per layer.

        For sequential operations (one per layer):
        Constraint: y_{o,e} ≤ 1 for each operation o, edge e

        This is automatically satisfied by binary variables, but we add it
        explicitly in case of parallel operations in the same layer.

        For now, we assume sequential operations, so this constraint is
        implicitly satisfied by the operation uniqueness constraint.

        Args:
            h: HiGHS model instance
        """
        # With sequential operations, each operation is its own layer
        # The binary constraint on y_{o,e} handles this automatically
        pass

    def _add_flow_conservation_constraints(self, h: highspy.Highs) -> None:
        """Add flow conservation constraints linking y and z variables.

        For each transition (qubit l, from_op o', to_op o):
            - Outflow from o': Σ_e z^l_{o, e', e} = y_{o', e'} for each e'
            - Inflow to o: Σ_{e'} z^l_{o, e', e} = y_{o, e} for each e

        Args:
            h: HiGHS model instance
        """
        for trans_idx, transition in enumerate(self.flow_transitions):
            from_op = transition.from_operation_index
            to_op = transition.to_operation_index

            # Outflow constraints: Σ_e z^l_{o, e', e} = y_{o', e'} for each e'
            for from_edge_idx in range(self.edge_count):
                indices = []
                values = []

                # Add z variables (positive coefficient)
                for to_edge_idx in range(self.edge_count):
                    z_idx = self._get_z_var_index(trans_idx, from_edge_idx, to_edge_idx)
                    indices.append(z_idx)
                    values.append(1.0)

                # Add y variable (negative coefficient for RHS)
                y_idx = self._get_y_var_index(from_op, from_edge_idx)
                indices.append(y_idx)
                values.append(-1.0)

                h.addRow(0.0, 0.0, len(indices), indices, values)

            # Inflow constraints: Σ_{e'} z^l_{o, e', e} = y_{o, e} for each e
            for to_edge_idx in range(self.edge_count):
                indices = []
                values = []

                # Add z variables (positive coefficient)
                for from_edge_idx in range(self.edge_count):
                    z_idx = self._get_z_var_index(trans_idx, from_edge_idx, to_edge_idx)
                    indices.append(z_idx)
                    values.append(1.0)

                # Add y variable (negative coefficient for RHS)
                y_idx = self._get_y_var_index(to_op, to_edge_idx)
                indices.append(y_idx)
                values.append(-1.0)

                h.addRow(0.0, 0.0, len(indices), indices, values)

    def _set_objective(self, h: highspy.Highs) -> None:
        """Set the objective function: minimise total distance.

        Objective: Σ dist(pos_l(e'), pos_l(e)) · z^l_{o, e', e}

        Args:
            h: HiGHS model instance
        """
        # Set objective sense to minimise
        h.changeObjectiveSense(highspy.ObjSense.kMinimize)

        # Set coefficients for z variables based on distances
        for trans_idx, transition in enumerate(self.flow_transitions):
            qubit = transition.qubit
            from_op = transition.from_operation_index
            to_op = transition.to_operation_index

            for from_edge_idx in range(self.edge_count):
                from_edge = self.edges[from_edge_idx]
                from_position = self._get_qubit_position_in_edge(
                    qubit, from_op, from_edge
                )

                for to_edge_idx in range(self.edge_count):
                    to_edge = self.edges[to_edge_idx]
                    to_position = self._get_qubit_position_in_edge(
                        qubit, to_op, to_edge
                    )

                    distance = self.distance_matrix[(from_position, to_position)]
                    z_idx = self._get_z_var_index(trans_idx, from_edge_idx, to_edge_idx)

                    h.changeColCost(z_idx, float(distance))

    def _extract_swaps_from_z_variables(
        self,
        col_values: list[float],
        operation_edge_assignment: dict[
            int, tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit]
        ],
    ) -> dict[int, list[quariadne.circuit.PhysicalSwap]]:
        """Extract swap sequence from z flow variables.

        Analyses z^l_{o,e',e} flow variables to determine qubit movements between
        consecutive operations. When a qubit's position changes between operations,
        computes the shortest path and generates corresponding SWAP operations.

        Args:
            col_values: Solution variable values from the LP solver
            operation_edge_assignment: Mapping from operation index to assigned edge

        Returns:
            Dictionary mapping operation index to list of PhysicalSwap objects.
            Swaps are keyed by the target operation index (swaps happen before that op).
        """
        inserted_swaps: dict[int, list[quariadne.circuit.PhysicalSwap]] = defaultdict(
            list
        )

        # Get undirected coupling for shortest path computation
        undirected_coupling = self.coupling_map.to_undirected()

        for trans_idx, transition in enumerate(self.flow_transitions):
            from_op = transition.from_operation_index
            to_op = transition.to_operation_index
            qubit = transition.qubit

            # Find which (e', e) pair has z > 0 for this transition
            for from_edge_idx in range(self.edge_count):
                for to_edge_idx in range(self.edge_count):
                    z_idx = self._get_z_var_index(trans_idx, from_edge_idx, to_edge_idx)
                    z_value = col_values[z_idx]

                    if z_value > 0.5:  # z variable is active
                        from_edge = self.edges[from_edge_idx]
                        to_edge = self.edges[to_edge_idx]

                        # Determine qubit positions at each operation
                        from_position = self._get_qubit_position_in_edge(
                            qubit, from_op, from_edge
                        )
                        to_position = self._get_qubit_position_in_edge(
                            qubit, to_op, to_edge
                        )

                        # If positions differ, compute swap path
                        if from_position != to_position:
                            # Compute shortest path between positions
                            try:
                                path = nx.shortest_path(
                                    undirected_coupling,
                                    from_position,
                                    to_position,
                                )
                            except nx.NetworkXNoPath:
                                continue

                            # Generate swaps along the path (stop one before target)
                            for i in range(len(path) - 1):
                                p_from = path[i]
                                p_to = path[i + 1]

                                swap = quariadne.circuit.PhysicalSwap(p_from, p_to)

                                # Check for duplicates
                                if swap not in inserted_swaps[to_op]:
                                    inserted_swaps[to_op].append(swap)

        return dict(inserted_swaps)

    def _extract_solution(self, h: highspy.Highs) -> BipartiteAllocationResult:
        """Extract the solution from the solved HiGHS model.

        Args:
            h: Solved HiGHS model instance

        Returns:
            BipartiteAllocationResult containing the solution
        """
        info = h.getInfo()
        objective_value = info.objective_function_value

        solution = h.getSolution()
        col_values = solution.col_value

        # Extract operation-edge assignments from y variables
        operation_edge_assignment: dict[
            int, tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit]
        ] = {}

        for op_idx in range(self.operation_count):
            for edge_idx in range(self.edge_count):
                y_idx = self._get_y_var_index(op_idx, edge_idx)
                if col_values[y_idx] > 0.5:  # Binary variable is 1
                    operation_edge_assignment[op_idx] = self.edges[edge_idx]
                    break

        # Build operation_to_edge_by_qubits for lookup by qubit pair
        # Key: (qubit1_idx, qubit2_idx, occurrence_count)
        operation_to_edge_by_qubits: dict[
            tuple[int, int, int],
            tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit],
        ] = {}
        qubit_pair_counts: dict[tuple[int, int], int] = {}

        for op_idx in range(self.operation_count):
            operation = self.operations[op_idx]
            q1, q2 = operation.qubits_participating
            key_pair = (q1.index, q2.index)

            occurrence = qubit_pair_counts.get(key_pair, 0)
            qubit_pair_counts[key_pair] = occurrence + 1

            full_key = (q1.index, q2.index, occurrence)
            if op_idx in operation_edge_assignment:
                operation_to_edge_by_qubits[full_key] = operation_edge_assignment[op_idx]

        # Derive initial mapping using bipartite matching for unique positions
        initial_mapping = self._derive_initial_mapping(operation_edge_assignment)

        # Extract swaps from z flow variables
        inserted_swaps = self._extract_swaps_from_z_variables(
            col_values, operation_edge_assignment
        )

        return BipartiteAllocationResult(
            objective_value=objective_value,
            operation_edge_assignment=operation_edge_assignment,
            operation_to_edge_by_qubits=operation_to_edge_by_qubits,
            initial_mapping=initial_mapping,
            operations=list(self.operations),
            inserted_swaps=inserted_swaps,
        )

    def _derive_initial_mapping(
        self,
        operation_edge_assignment: dict[
            int, tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit]
        ],
    ) -> dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]:
        """Derive initial qubit mapping from operation-edge assignments.

        Prioritises placing qubits at their first-operation positions. For qubits
        with conflicting positions, uses the earliest operation's position first.
        Remaining qubits (those not in any operation or with conflicts) are
        assigned to unused physical positions.

        Args:
            operation_edge_assignment: Mapping from operation index to assigned edge

        Returns:
            Initial logical-to-physical qubit mapping
        """
        initial_mapping: dict[
            quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit
        ] = {}
        used_positions: set[quariadne.circuit.PhysicalQubit] = set()

        # Collect (qubit, first_op_index, desired_position) tuples
        qubit_first_ops: list[
            tuple[quariadne.circuit.LogicalQubit, int, quariadne.circuit.PhysicalQubit]
        ] = []

        for qubit, op_info in self.qubit_operations.items():
            if op_info.operation_indices:
                first_op_idx = op_info.operation_indices[0]
                if first_op_idx in operation_edge_assignment:
                    edge = operation_edge_assignment[first_op_idx]
                    position = self._get_qubit_position_in_edge(
                        qubit, first_op_idx, edge
                    )
                    qubit_first_ops.append((qubit, first_op_idx, position))

        # Sort by first operation index (earlier operations get priority)
        qubit_first_ops.sort(key=lambda x: x[1])

        # Assign qubits to their desired positions (first-come, first-served)
        for qubit, first_op_idx, desired_pos in qubit_first_ops:
            if desired_pos not in used_positions:
                initial_mapping[qubit] = desired_pos
                used_positions.add(desired_pos)

        # For qubits with conflicts, find closest available position
        for qubit, first_op_idx, desired_pos in qubit_first_ops:
            if qubit not in initial_mapping:
                # Find nearest available position
                best_pos = None
                best_dist = float("inf")
                for pq in self.coupling_map.nodes:
                    if pq not in used_positions:
                        dist = self.distance_matrix.get((desired_pos, pq), float("inf"))
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = pq
                if best_pos is not None:
                    initial_mapping[qubit] = best_pos
                    used_positions.add(best_pos)

        # Assign remaining qubits (those not in any two-qubit operation)
        for qubit in self.routed_circuit.qubits:
            if qubit not in initial_mapping:
                for pq in self.coupling_map.nodes:
                    if pq not in used_positions:
                        initial_mapping[qubit] = pq
                        used_positions.add(pq)
                        break

        return initial_mapping

    def run(self) -> BipartiteAllocationResult:
        """Run the bipartite allocation LP optimisation.

        Returns:
            BipartiteAllocationResult containing the optimal solution
        """
        if self.operation_count == 0:
            # No two-qubit operations, return trivial mapping
            trivial_mapping = {
                qubit: quariadne.circuit.PhysicalQubit(qubit.index)
                for qubit in self.routed_circuit.qubits
            }
            return BipartiteAllocationResult(
                objective_value=0.0,
                operation_edge_assignment={},
                operation_to_edge_by_qubits={},
                initial_mapping=trivial_mapping,
                operations=[],
                inserted_swaps={},
            )

        # Build and solve the model
        h = self._build_model()
        h.run()

        # Check solution status
        model_status = h.getModelStatus()
        if model_status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"LP optimisation failed with status: {model_status}")

        return self._extract_solution(h)
