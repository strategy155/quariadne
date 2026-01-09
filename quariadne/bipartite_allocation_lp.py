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

    2. Position Exclusivity (per layer t, per position p):
       ``Σ_{o at t, e : p ∈ e} y_{o,e} ≤ 1``
       Ensures each physical position is used by at most one operation per layer.
       The constraint combines source and target usage into one constraint.

    3. Flow Conservation (Outflow): ``Σ_e z^l_{o,e',e} = y_{o',e'}`` for each e'.
       Total flow leaving edge e' must equal the assignment to that edge.

    4. Flow Conservation (Inflow): ``Σ_{e'} z^l_{o,e',e} = y_{o,e}`` for each e.
       Total flow entering edge e must equal the assignment to that edge.

**Objective Function:**
    Minimise ``Σ dist(pos_l(e'), pos_l(e)) · z^l_{o,e',e}``

    where ``pos_l(e)`` is the physical position where logical qubit l resides
    when assigned to edge e. The distance function returns the shortest path
    length on the undirected coupling graph (number of SWAPs needed).

LP Relaxation Properties
------------------------
Each constraint block has total unimodular (TU) structure individually:
operation uniqueness (assignment), position exclusivity (bipartite incidence),
and flow conservation (transportation). The combined matrix may yield fractional
y solutions; greedy argmax rounding selects the best edge per operation.

When y variables are integral, the z subproblem is a transportation problem
with guaranteed integral extreme points.

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
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import highspy
import networkx as nx
import numpy as np

import quariadne.benchmarks.constants
import quariadne.circuit
import quariadne.device_cache

_logger = logging.getLogger(__name__)

# =============================================================================
# Type Aliases
# =============================================================================
# Per https://docs.python.org/3/library/typing.html#type-aliases

type EdgeIndex = int
type OperationIndex = int
type TransitionIndex = int
type PhysicalEdge = tuple[
    quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit
]

# =============================================================================
# Named Constants
# =============================================================================
# Following milp.py style pattern for explicit constant definitions
# Ref: https://google.github.io/styleguide/pyguide.html#s2.14-named-arguments

# Constraint bound constants (matching milp.py naming convention)
EQUALITY_BOUND_ONE = 1.0
EQUALITY_BOUND_ZERO = 0.0

# Coefficient constants for constraint matrices
COEFFICIENT_POSITIVE = 1.0
COEFFICIENT_NEGATIVE = -1.0

# Variable bound constants for LP formulation
Y_VARIABLE_LOWER_BOUND = 0.0
Y_VARIABLE_UPPER_BOUND = 1.0
Z_VARIABLE_LOWER_BOUND = 0.0

# Default numpy dtype for constraint arrays
CONSTRAINT_ARRAY_DTYPE = np.float64
INDEX_ARRAY_DTYPE = np.int32


# =============================================================================
# Variable Type Enum and Registry Constants
# =============================================================================
# Following milp.py RoutingVariableType pattern


class BipartiteVariableType(Enum):
    """Enum for variable types in bipartite allocation LP.

    Attributes:
        Y_OPERATION_EDGE: Binary variable y_{o,e} indicating operation o assigned to edge e.
        Z_FLOW: Continuous flow variable z^l_{o,e',e} tracking qubit movement.
    """

    Y_OPERATION_EDGE = "y_operation_edge"
    Z_FLOW = "z_flow"


# Variable registry field constants (matching milp.py naming)
VARIABLE_COUNT = "count"
VARIABLE_OFFSET = "offset"


@dataclass
class BipartiteLPSolverOptions:
    """Configuration options for HiGHS LP solver in bipartite allocation.

    Centralises solver configuration following the HiGHSSolverOptions pattern
    from milp.py. Used to configure the HiGHS instance before solving.

    Attributes:
        output_flag: Whether to display solver output. Defaults to False.
        log_to_console: Whether to log to console. Defaults to False.
        solver: LP solver algorithm name. Uses HiPO with PARDISO by default.
        hipo_system_solver: System solver for HiPO. Uses PARDISO by default.
        threads: Number of threads for parallel solving.

    Ref: https://ergo-code.github.io/HiGHS/dev/options/definitions/
    Ref: https://github.com/strategy155/HiGHS (hipo-solvers branch)
    """

    output_flag: bool = False
    log_to_console: bool = False
    solver: str = quariadne.benchmarks.constants.HIPO_SOLVER_NAME
    hipo_system_solver: str = quariadne.benchmarks.constants.HIPO_SYSTEM_SOLVER
    threads: int = quariadne.benchmarks.constants.HIPO_DEFAULT_THREADS


if TYPE_CHECKING:
    import qiskit.transpiler


@functools.lru_cache(maxsize=8)
def _compute_all_pairs_distances(
    edges: frozenset[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Compute shortest path distances for a coupling graph.

    Checks device cache first for known backends, falls back to computation.

    Args:
        edges: Frozenset of (from_index, to_index) tuples defining the coupling graph.

    Returns:
        Dictionary mapping (source_index, target_index) to shortest path length.
    """
    # Check device cache first
    cached = quariadne.device_cache.get_topology_by_edges(edges)
    if cached is not None:
        return cached.distances

    # Fall back to computation for unknown topologies
    graph: nx.Graph[int] = nx.Graph()
    for from_index, to_index in edges:
        graph.add_edge(from_index, to_index)

    distances: dict[tuple[int, int], int] = {}
    all_pairs = dict(nx.all_pairs_shortest_path_length(graph))

    for source, targets in all_pairs.items():
        for target, length in targets.items():
            distances[(source, target)] = length

    return distances


# Type alias for path dictionary: maps (source, target) pair to list of node indices
ShortestPathsDict = dict[tuple[int, int], list[int]]


@functools.lru_cache(maxsize=8)
def _compute_all_pairs_shortest_paths(
    edges: frozenset[tuple[int, int]],
) -> ShortestPathsDict:
    """Compute shortest paths for all pairs of nodes in a coupling graph.

    Checks device cache first for known backends, falls back to computation.

    Args:
        edges: Frozenset of (from_index, to_index) tuples defining the coupling graph.

    Returns:
        Dictionary mapping (source_index, target_index) to list of node indices
        representing the shortest path from source to target (inclusive).
    """
    # Check device cache first
    cached = quariadne.device_cache.get_topology_by_edges(edges)
    if cached is not None:
        # Convert tuples to lists for compatibility
        return {key: list(path) for key, path in cached.paths.items()}

    # Fall back to computation for unknown topologies
    graph: nx.Graph[int] = nx.Graph()
    for from_index, to_index in edges:
        graph.add_edge(from_index, to_index)

    all_pairs_paths = dict(nx.all_pairs_shortest_path(graph))

    paths: ShortestPathsDict = {}
    for source_node, target_paths in all_pairs_paths.items():
        for target_node, path_nodes in target_paths.items():
            path_key = (source_node, target_node)
            paths[path_key] = path_nodes

    return paths


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

    Follows the pattern established by MilpRouterResult: stores solution data
    with swaps keyed by two-qubit operation index.

    Attributes:
        objective_value: The optimal objective value (total distance).
        operation_edge_assignment: Dict mapping operation index to assigned edge.
        operation_to_edge_by_qubits: Dict mapping (qubit1, qubit2, occurrence) to edge.
        initial_mapping: Initial logical-to-physical qubit mapping.
        final_mapping: Qubit mapping after all operations (for windowed solving).
        operations: List of operations in LP order for reference.
        inserted_swaps: Dict mapping two-qubit operation index to list of PhysicalSwap
            objects. Swaps for operation i should be inserted before that operation.
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
        fixed_operation_edges: dict[int, int] | None = None,
    ) -> None:
        """Initialise the bipartite allocation router.

        Args:
            coupling_map: NetworkX DiGraph representing physical qubit connectivity
            quantum_circuit: Abstract quantum circuit to be routed
            fixed_operation_edges: Optional dict mapping operation index to edge index.
                When provided, fixes the edge assignment for specified operations.
                Used for windowed solving where the shared layer's assignments from
                window k become constraints for window k+1.
        """
        self.coupling_map = coupling_map
        self.fixed_operation_edges = fixed_operation_edges
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

        # Build operation layers for position exclusivity constraints
        self.operation_layers = self._build_operation_layers()

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
                    distance_dict[(source_qubit, target_qubit)] = np.inf

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

        for operation_index, operation in enumerate(self.operations):
            for qubit in operation.qubits_participating:
                qubit_ops[qubit].operation_indices.append(operation_index)

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

    def _build_operation_layers(self) -> list[list[int]]:
        """Group operations into parallel layers based on qubit dependencies.

        Operations that share qubits cannot be in the same layer. Uses a greedy
        algorithm to assign each operation to the earliest possible layer.

        Returns:
            List of layers, where each layer is a list of operation indices.

        Reference:
            - https://docs.python.org/3/library/functions.html#max
            - https://docs.python.org/3/library/functions.html#enumerate
        """
        if self.operation_count == 0:
            return []

        layers: list[list[int]] = []
        # Track which layer each qubit was last used in
        qubit_last_layer: dict[quariadne.circuit.LogicalQubit, int] = {}

        for operation_index, operation in enumerate(self.operations):
            qubits = operation.qubits_participating

            # Find the earliest layer this operation can go in
            # Must be after any layer that uses the same qubits
            min_layer = 0
            for qubit in qubits:
                if qubit in qubit_last_layer:
                    min_layer = max(min_layer, qubit_last_layer[qubit] + 1)

            # Extend layers list if needed
            while len(layers) <= min_layer:
                layers.append([])

            # Add operation to this layer
            layers[min_layer].append(operation_index)

            # Update qubit_last_layer
            for qubit in qubits:
                qubit_last_layer[qubit] = min_layer

        return layers

    def _setup_variable_indexing(self) -> None:
        """Set up variable indexing for the LP formulation.

        Variables are laid out as:
            [y_{0,0}, y_{0,1}, ..., y_{O-1,E-1}, z^{l1}_{...}, z^{l2}_{...}, ...]

        where O is operation count and E is edge count.

        Also builds the variable_types registry following milp.py pattern.
        """
        # y variables: operation_count × edge_count
        self.y_var_count = self.operation_count * self.edge_count
        self.y_var_offset = 0

        # z variables: for each transition, edge_count × edge_count possibilities
        self.z_var_count = (
            len(self.flow_transitions) * self.edge_count * self.edge_count
        )
        self.z_var_offset = self.y_var_count

        self.total_var_count = self.y_var_count + self.z_var_count

        # Variable type registry (following milp.py pattern)
        self.variable_types: dict[BipartiteVariableType, dict[str, int]] = {
            BipartiteVariableType.Y_OPERATION_EDGE: {
                VARIABLE_COUNT: self.y_var_count,
                VARIABLE_OFFSET: self.y_var_offset,
            },
            BipartiteVariableType.Z_FLOW: {
                VARIABLE_COUNT: self.z_var_count,
                VARIABLE_OFFSET: self.z_var_offset,
            },
        }

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
        z_local_index = (
            transition_index * self.edge_count * self.edge_count
            + from_edge_index * self.edge_count
            + to_edge_index
        )
        return self.z_var_offset + z_local_index

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

    def _build_model(
        self, solver_options: BipartiteLPSolverOptions | None = None
    ) -> highspy.Highs:
        """Build the HiGHS LP model with all variables and constraints.

        Args:
            solver_options: Optional solver configuration. Uses defaults if None.

        Returns:
            Configured HiGHS model ready for optimisation.
        """
        if solver_options is None:
            solver_options = BipartiteLPSolverOptions()

        highs_model = highspy.Highs()
        highs_model.setOptionValue("output_flag", solver_options.output_flag)
        highs_model.setOptionValue("log_to_console", solver_options.log_to_console)

        # Configure HiPO solver with PARDISO for parallel sparse linear algebra
        highs_model.setOptionValue("solver", solver_options.solver)
        highs_model.setOptionValue(
            "hipo_system_solver", solver_options.hipo_system_solver
        )
        highs_model.setOptionValue("threads", solver_options.threads)

        # Add all variables
        self._add_variables(highs_model)

        # Add constraints
        self._add_operation_uniqueness_constraints(highs_model)
        self._add_edge_exclusivity_constraints(highs_model)
        self._add_flow_conservation_constraints(highs_model)

        # Set objective
        self._set_objective(highs_model)

        # Fix operations from shared layer (for windowed solving)
        self._add_fixed_operation_constraints(highs_model)

        return highs_model

    def _add_variables(self, highs_model: highspy.Highs) -> None:
        """Add all decision variables to the model.

        Uses batch HiGHS API for performance - adds all variables in one call
        instead of individual addVar() calls.

        Args:
            highs_model: HiGHS model instance

        Reference:
            - HiGHS Python API: https://ergo-code.github.io/HiGHS/dev/interfaces/python/
        """
        inf = highspy.kHighsInf

        # Pre-allocate arrays for batch variable addition
        lower_bounds = np.zeros(self.total_var_count, dtype=CONSTRAINT_ARRAY_DTYPE)
        upper_bounds = np.empty(self.total_var_count, dtype=CONSTRAINT_ARRAY_DTYPE)

        # y variables: bounds [0, 1] (binary semantics via LP relaxation)
        upper_bounds[: self.y_var_count] = Y_VARIABLE_UPPER_BOUND

        # z variables: bounds [0, inf] (continuous non-negative flow)
        upper_bounds[self.y_var_count :] = inf

        # Add all variables in one batch call
        highs_model.addVars(self.total_var_count, lower_bounds, upper_bounds)

        # NOTE: y variables are NOT set as integers despite bounds [0,1].
        # The constraint matrix is totally unimodular (TU), so the LP relaxation
        # yields integral solutions at vertices. This avoids MIP (NP-hard) and
        # uses pure LP (polynomial time).

    def _add_operation_uniqueness_constraints(self, highs_model: highspy.Highs) -> None:
        """Add constraints: each operation assigned to exactly one edge.

        Constraint: Σ_e y_{o,e} = 1 for each operation o

        Uses batch HiGHS API for performance - builds CSR matrix and adds all
        rows in one call instead of individual addRow() calls.

        Args:
            highs_model: HiGHS model instance

        Reference:
            - HiGHS Python API: https://ergo-code.github.io/HiGHS/dev/interfaces/python/
        """
        num_constraints = self.operation_count
        nonzeros_per_constraint = self.edge_count
        total_nonzeros = num_constraints * nonzeros_per_constraint

        # Pre-allocate CSR format arrays
        lower_bounds = np.full(
            num_constraints, EQUALITY_BOUND_ONE, dtype=CONSTRAINT_ARRAY_DTYPE
        )
        upper_bounds = np.full(
            num_constraints, EQUALITY_BOUND_ONE, dtype=CONSTRAINT_ARRAY_DTYPE
        )
        row_starts = np.arange(
            0, total_nonzeros + 1, nonzeros_per_constraint, dtype=INDEX_ARRAY_DTYPE
        )
        col_indices = np.empty(total_nonzeros, dtype=INDEX_ARRAY_DTYPE)
        values = np.full(
            total_nonzeros, COEFFICIENT_POSITIVE, dtype=CONSTRAINT_ARRAY_DTYPE
        )

        # Fill column indices: for each operation, include all edge y variables
        for operation_index in range(self.operation_count):
            start_index = operation_index * self.edge_count
            for edge_index in range(self.edge_count):
                col_indices[start_index + edge_index] = self._get_y_var_index(
                    operation_index, edge_index
                )

        highs_model.addRows(
            num_constraints,
            lower_bounds,
            upper_bounds,
            total_nonzeros,
            row_starts,
            col_indices,
            values,
        )

    def _add_edge_exclusivity_constraints(self, highs_model: highspy.Highs) -> None:
        """Add constraints: each physical position used at most once per layer.

        For each layer t and physical position p:
            Σ_{o at t, e : p ∈ e} y_{o,e} ≤ 1

        Where p ∈ e means p is either the source or target of edge e.

        This ensures physical position exclusivity: at each time step, each
        physical qubit position is used by at most one operation. A single
        combined constraint per position prevents conflicts where one operation
        uses a position as source and another uses it as target.

        Args:
            highs_model: HiGHS model instance
        """
        # Build index mapping: position -> list of edge indices touching that position
        position_to_edge_indices: dict[quariadne.circuit.PhysicalQubit, list[int]] = {
            node: [] for node in self.coupling_map.nodes
        }

        for edge_index, edge in enumerate(self.edges):
            source, target = edge
            position_to_edge_indices[source].append(edge_index)
            position_to_edge_indices[target].append(edge_index)

        # Filter layers with multiple operations (single-operation layers are trivially satisfied)
        multi_operation_layers = [
            layer for layer in self.operation_layers if len(layer) > 1
        ]

        # Add constraints for each layer and position
        for layer in multi_operation_layers:
            for position, edge_indices in position_to_edge_indices.items():
                # Combined position constraint: Σ_{o at t, e : p ∈ e} y_{o,e} ≤ 1
                constraint_indices = []
                constraint_values = []

                for operation_index in layer:
                    for edge_index in edge_indices:
                        variable_index = self._get_y_var_index(
                            operation_index, edge_index
                        )
                        constraint_indices.append(variable_index)
                        constraint_values.append(COEFFICIENT_POSITIVE)

                if len(constraint_indices) > 1:
                    highs_model.addRow(
                        -highspy.kHighsInf,
                        EQUALITY_BOUND_ONE,
                        len(constraint_indices),
                        constraint_indices,
                        constraint_values,
                    )

    def _add_flow_conservation_constraints(self, highs_model: highspy.Highs) -> None:
        """Add flow conservation constraints linking y and z variables.

        For each transition (qubit l, from_op o', to_op o):
            - Outflow from o': Σ_e z^l_{o, e', e} = y_{o', e'} for each e'
            - Inflow to o: Σ_{e'} z^l_{o, e', e} = y_{o, e} for each e

        Uses batch HiGHS API for performance - builds CSR matrix and adds all rows
        in one call instead of individual addRow() calls.

        Args:
            highs_model: HiGHS model instance

        Reference:
            - HiGHS Python API: https://ergo-code.github.io/HiGHS/dev/interfaces/python/
        """
        num_transitions = len(self.flow_transitions)
        num_constraints = num_transitions * 2 * self.edge_count
        nonzeros_per_constraint = self.edge_count + 1
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

        for transition_index, transition in enumerate(self.flow_transitions):
            from_operation_index = transition.from_operation_index
            to_operation_index = transition.to_operation_index

            # Outflow constraints
            for from_edge_index in range(self.edge_count):
                row_starts[row_index] = nonzero_index

                for to_edge_index in range(self.edge_count):
                    z_variable_index = self._get_z_var_index(
                        transition_index, from_edge_index, to_edge_index
                    )
                    col_indices[nonzero_index] = z_variable_index
                    values[nonzero_index] = COEFFICIENT_POSITIVE
                    nonzero_index += 1

                y_variable_index = self._get_y_var_index(
                    from_operation_index, from_edge_index
                )
                col_indices[nonzero_index] = y_variable_index
                values[nonzero_index] = COEFFICIENT_NEGATIVE
                nonzero_index += 1
                row_index += 1

            # Inflow constraints
            for to_edge_index in range(self.edge_count):
                row_starts[row_index] = nonzero_index

                for from_edge_index in range(self.edge_count):
                    z_variable_index = self._get_z_var_index(
                        transition_index, from_edge_index, to_edge_index
                    )
                    col_indices[nonzero_index] = z_variable_index
                    values[nonzero_index] = COEFFICIENT_POSITIVE
                    nonzero_index += 1

                y_variable_index = self._get_y_var_index(
                    to_operation_index, to_edge_index
                )
                col_indices[nonzero_index] = y_variable_index
                values[nonzero_index] = COEFFICIENT_NEGATIVE
                nonzero_index += 1
                row_index += 1

        row_starts[num_constraints] = total_nonzeros

        highs_model.addRows(
            num_constraints,
            lower_bounds,
            upper_bounds,
            total_nonzeros,
            row_starts,
            col_indices,
            values,
        )

    def _set_objective(self, highs_model: highspy.Highs) -> None:
        """Set the objective function: minimise total distance.

        Objective: Σ dist(pos_l(e'), pos_l(e)) · z^l_{o, e', e}

        Uses batch HiGHS API for performance - sets all z variable costs in one call
        instead of individual changeColCost() calls.

        Args:
            highs_model: HiGHS model instance

        Reference:
            - HiGHS Python API: https://ergo-code.github.io/HiGHS/dev/interfaces/python/
        """
        # Set objective sense to minimise
        highs_model.changeObjectiveSense(highspy.ObjSense.kMinimize)

        # Pre-allocate arrays for batch API call
        z_indices = np.empty(self.z_var_count, dtype=INDEX_ARRAY_DTYPE)
        z_costs = np.empty(self.z_var_count, dtype=CONSTRAINT_ARRAY_DTYPE)

        # Compute all z variable objective coefficients
        variable_index = 0
        for transition_index, transition in enumerate(self.flow_transitions):
            qubit = transition.qubit
            from_operation_index = transition.from_operation_index
            to_operation_index = transition.to_operation_index

            for from_edge_index in range(self.edge_count):
                from_edge = self.edges[from_edge_index]
                from_position = self._get_qubit_position_in_edge(
                    qubit, from_operation_index, from_edge
                )

                for to_edge_index in range(self.edge_count):
                    to_edge = self.edges[to_edge_index]
                    to_position = self._get_qubit_position_in_edge(
                        qubit, to_operation_index, to_edge
                    )

                    distance = self.distance_matrix[(from_position, to_position)]
                    z_variable_index = self._get_z_var_index(
                        transition_index, from_edge_index, to_edge_index
                    )

                    z_indices[variable_index] = z_variable_index
                    z_costs[variable_index] = float(distance)
                    variable_index += 1

        # Set all z variable costs in one batch call
        highs_model.changeColsCost(self.z_var_count, z_indices, z_costs)

    def _add_fixed_operation_constraints(self, highs_model: highspy.Highs) -> None:
        """Fix y variables for operations with pre-assigned edges.

        When fixed_operation_edges is provided (for windowed solving), fixes the
        edge assignment for specified operations. The shared layer's assignments
        from the previous window become constraints for the current window.

        For each fixed operation:
        - Set y_{op, assigned_edge} bounds to [1, 1] (force to 1)
        - Set y_{op, other_edges} bounds to [0, 0] (force to 0)

        Args:
            highs_model: HiGHS model instance

        Reference:
            - HiGHS Python API: https://ergo-code.github.io/HiGHS/dev/interfaces/python/
        """
        if self.fixed_operation_edges is None:
            return

        for operation_index, assigned_edge_index in self.fixed_operation_edges.items():
            for edge_index in range(self.edge_count):
                y_variable_index = self._get_y_var_index(operation_index, edge_index)
                if edge_index == assigned_edge_index:
                    highs_model.changeColBounds(
                        y_variable_index,
                        Y_VARIABLE_UPPER_BOUND,
                        Y_VARIABLE_UPPER_BOUND,
                    )
                else:
                    highs_model.changeColBounds(
                        y_variable_index,
                        Y_VARIABLE_LOWER_BOUND,
                        Y_VARIABLE_LOWER_BOUND,
                    )

    def _find_best_edge_pair_for_transition(
        self,
        transition_index: int,
        col_values: list[float],
    ) -> tuple[int, int]:
        """Find the (source_edge, target_edge) pair with maximum z value for a transition.

        For the given flow transition, finds the (e', e) edge pair with the highest
        z variable value using numpy argmax. This is more robust than a simple
        threshold for handling fractional LP solutions.

        Args:
            transition_index: Index of the flow transition in self.flow_transitions.
            col_values: Solution variable values from the LP solver.

        Returns:
            Tuple of (source_edge_index, target_edge_index) for the best edge pair.

        Reference:
            - https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        """
        # Get the start index for this transition's z variables
        z_block_start_index = self._get_z_var_index(transition_index, 0, 0)
        z_block_size = self.edge_count * self.edge_count

        # Extract z values for this transition and find argmax
        z_block = np.array(
            col_values[z_block_start_index : z_block_start_index + z_block_size]
        )
        best_flat_index = np.argmax(z_block)

        # Convert flat index to (from_edge, to_edge) pair
        source_edge_index, target_edge_index = divmod(best_flat_index, self.edge_count)
        return int(source_edge_index), int(target_edge_index)

    def _extract_swaps_from_z_variables(
        self,
        col_values: list[float],
        operation_edge_assignment: dict[
            int, tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit]
        ],
    ) -> dict[int, list[quariadne.circuit.PhysicalSwap]]:
        """Extract swap sequence from z flow variables.

        Analyses z^l_{o,e',e} flow variables to determine qubit movements between
        consecutive operations. For each transition, selects the (e', e) pair with
        maximum z value (robust to fractional LP solutions), then generates SWAP
        operations along the shortest path between positions.

        Swaps are keyed by the target operation index (the operation BEFORE which
        the swaps should be inserted), matching MilpRouterResult.inserted_swaps.

        Args:
            col_values: Solution variable values from the LP solver.
            operation_edge_assignment: Mapping from operation index to assigned edge.

        Returns:
            Dictionary mapping operation index to list of PhysicalSwap objects.

        Raises:
            RuntimeError: If no path exists between two physical qubit positions.
        """
        inserted_swaps: dict[int, list[quariadne.circuit.PhysicalSwap]] = defaultdict(
            list
        )

        # Get cached shortest paths using the same edge set used for distance computation
        coupling_edges = frozenset(
            (edge[0].index, edge[1].index) for edge in self.coupling_map.edges()
        )
        cached_paths = _compute_all_pairs_shortest_paths(coupling_edges)

        # Process each flow transition to extract swaps
        for transition_index, transition in enumerate(self.flow_transitions):
            source_operation = transition.from_operation_index
            target_operation = transition.to_operation_index
            logical_qubit = transition.qubit

            # Find the edge pair with maximum z value for this transition
            source_edge_index, target_edge_index = (
                self._find_best_edge_pair_for_transition(transition_index, col_values)
            )

            # Get the actual edges from indices
            source_edge = self.edges[source_edge_index]
            target_edge = self.edges[target_edge_index]

            # Determine where the qubit sits at source and target operations
            source_physical_position = self._get_qubit_position_in_edge(
                logical_qubit, source_operation, source_edge
            )
            target_physical_position = self._get_qubit_position_in_edge(
                logical_qubit, target_operation, target_edge
            )

            # If positions differ, generate swap path
            if source_physical_position != target_physical_position:
                path_key = (
                    source_physical_position.index,
                    target_physical_position.index,
                )

                # Verify path exists in cached paths
                if path_key not in cached_paths:
                    raise RuntimeError(
                        f"No path found between physical qubits "
                        f"{source_physical_position} and {target_physical_position}."
                    )

                # Get path as list of node indices and generate SWAPs
                swap_path_indices = cached_paths[path_key]
                swap_path_length = len(swap_path_indices)

                for path_step_index in range(swap_path_length - 1):
                    current_qubit_index = swap_path_indices[path_step_index]
                    next_qubit_index = swap_path_indices[path_step_index + 1]

                    current_physical_qubit = quariadne.circuit.PhysicalQubit(
                        current_qubit_index
                    )
                    next_physical_qubit = quariadne.circuit.PhysicalQubit(
                        next_qubit_index
                    )

                    swap = quariadne.circuit.PhysicalSwap(
                        current_physical_qubit, next_physical_qubit
                    )

                    # Add swap if not already present (avoid duplicates)
                    current_operation_swaps = inserted_swaps[target_operation]
                    if swap not in current_operation_swaps:
                        current_operation_swaps.append(swap)

        return dict(inserted_swaps)

    def _extract_solution(
        self, highs_model: highspy.Highs
    ) -> BipartiteAllocationResult:
        """Extract the solution from the solved HiGHS model.

        Args:
            highs_model: Solved HiGHS model instance

        Returns:
            BipartiteAllocationResult containing the solution
        """
        info = highs_model.getInfo()
        objective_value = info.objective_function_value

        solution = highs_model.getSolution()
        col_values = solution.col_value

        # Extract operation-edge assignments from y variables using greedy rounding.
        # For LP relaxation, y values are fractional. We pick the edge with
        # the highest y value for each operation (greedy argmax rounding).
        # Uses numpy for vectorised argmax instead of Python loops.
        y_values = np.array(col_values[: self.y_var_count]).reshape(
            self.operation_count, self.edge_count
        )
        best_edge_indices = np.argmax(y_values, axis=1)

        operation_edge_assignment: dict[
            int, tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit]
        ] = {
            operation_index: self.edges[best_edge_indices[operation_index]]
            for operation_index in range(self.operation_count)
        }

        # Build operation_to_edge_by_qubits for lookup by qubit pair
        # Key: (left_qubit_index, right_qubit_index, occurrence_count)
        operation_to_edge_by_qubits: dict[
            tuple[int, int, int],
            tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit],
        ] = {}
        qubit_pair_counts: dict[tuple[int, int], int] = {}

        for operation_index in range(self.operation_count):
            operation = self.operations[operation_index]
            left_qubit, right_qubit = operation.qubits_participating
            qubit_pair_key = (left_qubit.index, right_qubit.index)

            occurrence = qubit_pair_counts.get(qubit_pair_key, 0)
            qubit_pair_counts[qubit_pair_key] = occurrence + 1

            full_key = (left_qubit.index, right_qubit.index, occurrence)
            if operation_index in operation_edge_assignment:
                operation_to_edge_by_qubits[full_key] = operation_edge_assignment[
                    operation_index
                ]

        # Derive initial mapping using bipartite matching for unique positions
        initial_mapping = self._derive_initial_mapping(operation_edge_assignment)

        # Extract final mapping for windowed solving
        final_mapping = self._extract_final_mapping(operation_edge_assignment)

        # Extract swaps from z flow variables
        inserted_swaps = self._extract_swaps_from_z_variables(
            col_values, operation_edge_assignment
        )

        return BipartiteAllocationResult(
            objective_value=objective_value,
            operation_edge_assignment=operation_edge_assignment,
            operation_to_edge_by_qubits=operation_to_edge_by_qubits,
            initial_mapping=initial_mapping,
            final_mapping=final_mapping,
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

        # Collect (qubit, first_operation_index, desired_position) tuples
        qubit_first_operations: list[
            tuple[quariadne.circuit.LogicalQubit, int, quariadne.circuit.PhysicalQubit]
        ] = []

        for qubit, operation_info in self.qubit_operations.items():
            if operation_info.operation_indices:
                first_operation_index = operation_info.operation_indices[0]
                if first_operation_index in operation_edge_assignment:
                    edge = operation_edge_assignment[first_operation_index]
                    position = self._get_qubit_position_in_edge(
                        qubit, first_operation_index, edge
                    )
                    qubit_first_operations.append(
                        (qubit, first_operation_index, position)
                    )

        # Sort by first operation index (earlier operations get priority)
        qubit_first_operations.sort(key=lambda x: x[1])

        # Assign qubits to their desired positions (first-come, first-served)
        for qubit, first_operation_index, desired_position in qubit_first_operations:
            if desired_position not in used_positions:
                initial_mapping[qubit] = desired_position
                used_positions.add(desired_position)

        # For qubits with conflicts, find closest available position
        for qubit, first_operation_index, desired_position in qubit_first_operations:
            if qubit not in initial_mapping:
                # Find nearest available position
                best_position = None
                best_distance = np.inf
                for physical_qubit in self.coupling_map.nodes:
                    if physical_qubit not in used_positions:
                        distance = self.distance_matrix.get(
                            (desired_position, physical_qubit), np.inf
                        )
                        if distance < best_distance:
                            best_distance = distance
                            best_position = physical_qubit
                if best_position is not None:
                    initial_mapping[qubit] = best_position
                    used_positions.add(best_position)

        # Assign remaining qubits (those not in any two-qubit operation)
        for qubit in self.routed_circuit.qubits:
            if qubit not in initial_mapping:
                for physical_qubit in self.coupling_map.nodes:
                    if physical_qubit not in used_positions:
                        initial_mapping[qubit] = physical_qubit
                        used_positions.add(physical_qubit)
                        break

        return initial_mapping

    def _extract_final_mapping(
        self,
        operation_edge_assignment: dict[
            int, tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit]
        ],
    ) -> dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]:
        """Extract qubit positions after all operations.

        For each qubit involved in two-qubit operations, finds its position at
        its last operation. Used for windowed solving where the final positions
        become constraints for the next window.

        Args:
            operation_edge_assignment: Mapping from operation index to assigned edge

        Returns:
            Final logical-to-physical qubit mapping
        """
        final_mapping: dict[
            quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit
        ] = {}

        for qubit, operation_info in self.qubit_operations.items():
            if not operation_info.operation_indices:
                continue

            last_operation_index = operation_info.operation_indices[-1]
            if last_operation_index not in operation_edge_assignment:
                continue

            edge = operation_edge_assignment[last_operation_index]
            operation = self.operations[last_operation_index]
            left_qubit, right_qubit = operation.qubits_participating

            if qubit == left_qubit:
                final_mapping[qubit] = edge[0]
            elif qubit == right_qubit:
                final_mapping[qubit] = edge[1]

        return final_mapping

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
                objective_value=EQUALITY_BOUND_ZERO,
                operation_edge_assignment={},
                operation_to_edge_by_qubits={},
                initial_mapping=trivial_mapping,
                final_mapping=trivial_mapping,
                operations=[],
                inserted_swaps={},
            )

        # Build and solve the model
        t0 = time.perf_counter()
        highs_model = self._build_model()
        t1 = time.perf_counter()
        _logger.info(
            "Model built: %d vars, %d constraints in %.2fs",
            highs_model.getNumCol(),
            highs_model.getNumRow(),
            t1 - t0,
        )

        highs_model.run()
        t2 = time.perf_counter()
        _logger.info("LP solved in %.2fs", t2 - t1)

        # Check solution status
        model_status = highs_model.getModelStatus()
        if model_status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"LP optimisation failed with status: {model_status}")

        result = self._extract_solution(highs_model)
        t3 = time.perf_counter()
        _logger.info("Solution extracted in %.2fs", t3 - t2)

        return result


class WindowedBipartiteRouter(BipartiteAllocationRouter):
    """Windowed version of BipartiteAllocationRouter.

    Solves large circuits in overlapping windows of N layers. Each window
    shares its last layer with the next window's first layer for continuity.
    """

    def __init__(
        self,
        coupling_map: nx.DiGraph,
        quantum_circuit: quariadne.circuit.AbstractQuantumCircuit,
        window_size: int = 5,
    ) -> None:
        """Initialise windowed router."""
        super().__init__(coupling_map, quantum_circuit)
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.window_size = window_size

    def _compute_windows(self) -> list[tuple[int, int]]:
        """Compute overlapping window boundaries.

        Returns:
            List of (start_layer, end_layer) tuples
        """
        n = len(self.operation_layers)
        if n == 0:
            return []

        windows = []
        start = 0
        while start < n:
            end = min(start + self.window_size, n)
            windows.append((start, end))
            if end >= n:
                break
            start = end - 1  # 1-layer overlap
        return windows

    def _create_subcircuit(
        self, operation_indices: list[int]
    ) -> quariadne.circuit.AbstractQuantumCircuit:
        """Create a sub-circuit containing only the specified operations.

        Constructs a new AbstractQuantumCircuit with the same qubits as the
        original circuit but containing only the operations at the given indices.
        Used to partition the circuit into windows for separate LP solving.

        Args:
            operation_indices: List of indices into self.operations specifying
                which two-qubit operations to include in the sub-circuit.

        Returns:
            A new AbstractQuantumCircuit containing only the selected operations,
            preserving the original qubit set.

        Reference:
            - https://docs.python.org/3/library/stdtypes.html#list
        """
        selected_operations = [
            self.operations[operation_index] for operation_index in operation_indices
        ]
        return quariadne.circuit.AbstractQuantumCircuit(
            qubits=self.routed_circuit.qubits,
            operations=selected_operations,
        )

    def _build_global_to_local_index_map(
        self, window_operation_indices: list[int]
    ) -> dict[int, int]:
        """Build a mapping from global operation indices to window-local indices.

        When solving a window, operations are re-indexed starting from 0 within
        the window's sub-circuit. This mapping allows conversion between the
        global indices (in self.operations) and local indices (in the window).

        Args:
            window_operation_indices: List of global operation indices that appear
                in this window, in the order they appear in the sub-circuit.

        Returns:
            Dictionary mapping global_index -> local_index. For example, if
            window_operation_indices=[5, 8, 12], returns {5: 0, 8: 1, 12: 2}.
        """
        global_to_local_map = {}
        for local_index, global_index in enumerate(window_operation_indices):
            global_to_local_map[global_index] = local_index
        return global_to_local_map

    def _convert_constraints_to_local_indices(
        self,
        global_fixed_edges: dict[int, int] | None,
        global_to_local_map: dict[int, int],
    ) -> dict[int, int] | None:
        """Convert edge constraints from global to window-local operation indices.

        The shared layer's operation-edge assignments from the previous window
        are expressed in global indices. This method converts them to the local
        indices used by the current window's BipartiteAllocationRouter.

        Args:
            global_fixed_edges: Constraints from previous window mapping
                global_operation_index -> edge_index. None if this is the first window.
            global_to_local_map: Mapping from global to local indices for this window.

        Returns:
            Dictionary mapping local_operation_index -> edge_index for operations
            that exist in both the constraints and the current window. Returns None
            if no constraints apply to this window.
        """
        if global_fixed_edges is None:
            return None

        local_fixed_edges = {}
        for global_operation_index, edge_index in global_fixed_edges.items():
            if global_operation_index in global_to_local_map:
                local_operation_index = global_to_local_map[global_operation_index]
                local_fixed_edges[local_operation_index] = edge_index

        return local_fixed_edges if local_fixed_edges else None

    def _solve_window(
        self,
        window_operation_indices: list[int],
        global_fixed_edges: dict[int, int] | None,
    ) -> tuple[BipartiteAllocationResult, BipartiteAllocationRouter, dict[int, int]]:
        """Solve the LP for a single window of operations.

        Creates a sub-circuit containing only the window's operations, converts
        any fixed constraints from global to local indices, and runs a fresh
        BipartiteAllocationRouter on the sub-circuit.

        Args:
            window_operation_indices: Global operation indices in this window.
            global_fixed_edges: Edge constraints from previous window's shared layer,
                mapping global_operation_index -> edge_index. None for first window.

        Returns:
            Tuple containing:
                - BipartiteAllocationResult from solving the window
                - BipartiteAllocationRouter instance (for accessing edge_to_index)
                - global_to_local_map for converting results back to global indices
        """
        global_to_local_map = self._build_global_to_local_index_map(
            window_operation_indices
        )

        local_fixed_edges = self._convert_constraints_to_local_indices(
            global_fixed_edges, global_to_local_map
        )

        window_circuit = self._create_subcircuit(window_operation_indices)

        window_router = BipartiteAllocationRouter(
            self.coupling_map,
            window_circuit,
            fixed_operation_edges=local_fixed_edges,
        )

        window_result = window_router.run()

        return window_result, window_router, global_to_local_map

    def _extract_shared_layer_constraints(
        self,
        shared_layer_index: int,
        window_result: BipartiteAllocationResult,
        window_router: BipartiteAllocationRouter,
        global_to_local_map: dict[int, int],
    ) -> dict[int, int]:
        """Extract edge constraints from the shared layer for the next window.

        The shared layer (last layer of current window) becomes the first layer
        of the next window. This method extracts the operation-edge assignments
        that will be fixed as constraints in the next window's LP.

        Args:
            shared_layer_index: Index of the shared layer in self.operation_layers.
            window_result: Result from solving the current window.
            window_router: Router instance used for the current window.
            global_to_local_map: Mapping from global to local indices.

        Returns:
            Dictionary mapping global_operation_index -> edge_index for all
            operations in the shared layer.
        """
        # Get all operations in the shared layer (using global indices)
        shared_layer_global_ops = self.operation_layers[shared_layer_index]
        constraints = {}

        for global_operation_index in shared_layer_global_ops:
            # Check if this operation was part of the window we just solved
            is_in_window = global_operation_index in global_to_local_map

            if is_in_window:
                # Convert to local index to look up in window result
                local_operation_index = global_to_local_map[global_operation_index]

                # Verify the operation has an edge assignment in the result
                has_assignment = (
                    local_operation_index in window_result.operation_edge_assignment
                )

                if has_assignment:
                    # Get the assigned edge (as PhysicalQubit tuple)
                    assigned_edge = window_result.operation_edge_assignment[
                        local_operation_index
                    ]

                    # Convert edge tuple to edge index for constraint storage
                    edge_index = window_router.edge_to_index[assigned_edge]

                    # Store constraint using global operation index
                    constraints[global_operation_index] = edge_index

        return constraints

    def _accumulate_window_results(
        self,
        window_result: BipartiteAllocationResult,
        window_operation_indices: list[int],
        accumulated_edge_assignments: dict[
            int, tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit]
        ],
        accumulated_swaps: dict[int, list[quariadne.circuit.PhysicalSwap]],
    ) -> None:
        """Accumulate results from a window into the global result containers.

        Converts window-local indices back to global indices and merges the
        edge assignments and swap sequences into the accumulated containers.

        Args:
            window_result: Result from solving a single window.
            window_operation_indices: Global operation indices in this window.
            accumulated_edge_assignments: Global container for edge assignments (modified).
            accumulated_swaps: Global container for swap sequences (modified).
        """
        # Convert local edge assignments to global indices
        for (
            local_operation_index,
            assigned_edge,
        ) in window_result.operation_edge_assignment.items():
            global_operation_index = window_operation_indices[local_operation_index]
            accumulated_edge_assignments[global_operation_index] = assigned_edge

        # Convert local swap sequences to global indices
        for (
            local_operation_index,
            swap_list,
        ) in window_result.inserted_swaps.items():
            global_operation_index = window_operation_indices[local_operation_index]

            # Initialise list if first swaps for this operation
            if global_operation_index not in accumulated_swaps:
                accumulated_swaps[global_operation_index] = []

            # Extend with swaps from this window
            accumulated_swaps[global_operation_index].extend(swap_list)

    def _build_final_result(
        self,
        total_objective: float,
        accumulated_edge_assignments: dict[
            int, tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit]
        ],
        accumulated_swaps: dict[int, list[quariadne.circuit.PhysicalSwap]],
        initial_mapping: dict[
            quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit
        ],
    ) -> BipartiteAllocationResult:
        """Build the final BipartiteAllocationResult from accumulated window results.

        Constructs the operation_to_edge_by_qubits lookup table and final_mapping
        from the accumulated edge assignments across all windows.

        Args:
            total_objective: Sum of objective values from all windows.
            accumulated_edge_assignments: Merged edge assignments (global indices).
            accumulated_swaps: Merged swap sequences (global indices).
            initial_mapping: Initial mapping from the first window.

        Returns:
            Complete BipartiteAllocationResult for the entire circuit.
        """
        # Build operation_to_edge_by_qubits lookup table
        operation_to_edge_by_qubits: dict[
            tuple[int, int, int],
            tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit],
        ] = {}
        qubit_pair_occurrence_counts: dict[tuple[int, int], int] = {}

        for operation_index, operation in enumerate(self.operations):
            qubit_left, qubit_right = operation.qubits_participating
            qubit_pair_key = (qubit_left.index, qubit_right.index)

            # Track occurrence count for this qubit pair
            occurrence = qubit_pair_occurrence_counts.get(qubit_pair_key, 0)
            qubit_pair_occurrence_counts[qubit_pair_key] = occurrence + 1

            # Store edge assignment with (qubit1, qubit2, occurrence) key
            if operation_index in accumulated_edge_assignments:
                full_key = (qubit_left.index, qubit_right.index, occurrence)
                operation_to_edge_by_qubits[full_key] = accumulated_edge_assignments[
                    operation_index
                ]

        # Build final mapping from last operations of each qubit
        final_mapping: dict[
            quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit
        ] = {}

        # Process operations in reverse to find last assignment for each qubit
        for operation_index in reversed(range(len(self.operations))):
            if operation_index in accumulated_edge_assignments:
                operation = self.operations[operation_index]
                assigned_edge = accumulated_edge_assignments[operation_index]
                qubit_left, qubit_right = operation.qubits_participating
                edge_left, edge_right = assigned_edge

                # Assign qubit to edge endpoint if not already assigned
                if qubit_left not in final_mapping:
                    final_mapping[qubit_left] = edge_left
                if qubit_right not in final_mapping:
                    final_mapping[qubit_right] = edge_right

        return BipartiteAllocationResult(
            objective_value=total_objective,
            operation_edge_assignment=accumulated_edge_assignments,
            operation_to_edge_by_qubits=operation_to_edge_by_qubits,
            initial_mapping=initial_mapping,
            final_mapping=final_mapping,
            operations=list(self.operations),
            inserted_swaps=accumulated_swaps,
        )

    def run(self) -> BipartiteAllocationResult:
        """Run windowed bipartite allocation optimisation.

        Overrides the parent run() method to solve large circuits in overlapping
        windows. For small circuits (fewer layers than window_size), delegates
        to the parent's single-solve approach.

        Returns:
            BipartiteAllocationResult containing the merged solution from all windows.
        """
        # For small circuits, use the parent's single-solve approach
        num_layers = len(self.operation_layers)
        if num_layers <= self.window_size:
            _logger.info("Circuit has %d layers, using single solve", num_layers)
            return super().run()

        # Compute overlapping windows
        windows = self._compute_windows()
        _logger.info(
            "Windowed solving: %d layers, %d windows (size=%d)",
            num_layers,
            len(windows),
            self.window_size,
        )

        # Initialise accumulators for results across windows
        accumulated_edge_assignments: dict[
            int, tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit]
        ] = {}
        accumulated_swaps: dict[int, list[quariadne.circuit.PhysicalSwap]] = {}
        total_objective = 0.0
        initial_mapping = None

        # Constraints from previous window's shared layer
        global_fixed_edges: dict[int, int] | None = None

        # Solve each window sequentially
        for window_index, (start_layer, end_layer) in enumerate(windows):
            _logger.info(
                "Window %d/%d: layers [%d, %d)",
                window_index + 1,
                len(windows),
                start_layer,
                end_layer,
            )

            # Collect all operation indices in this window's layers
            window_operation_indices = []
            for layer_index in range(start_layer, end_layer):
                window_operation_indices.extend(self.operation_layers[layer_index])

            # Solve this window
            window_result, window_router, global_to_local_map = self._solve_window(
                window_operation_indices, global_fixed_edges
            )

            # Accumulate objective value
            total_objective += window_result.objective_value

            # Store initial mapping from first window only
            if initial_mapping is None:
                initial_mapping = window_result.initial_mapping

            # Merge window results into accumulators
            self._accumulate_window_results(
                window_result,
                window_operation_indices,
                accumulated_edge_assignments,
                accumulated_swaps,
            )

            # Extract constraints for next window from shared layer
            is_not_last_window = end_layer < num_layers
            if is_not_last_window:
                shared_layer_index = end_layer - 1
                global_fixed_edges = self._extract_shared_layer_constraints(
                    shared_layer_index,
                    window_result,
                    window_router,
                    global_to_local_map,
                )

        # Build and return final result
        return self._build_final_result(
            total_objective,
            accumulated_edge_assignments,
            accumulated_swaps,
            initial_mapping or {},
        )
