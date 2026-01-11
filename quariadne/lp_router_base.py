"""Base class for LP-based quantum circuit routers.

This module provides the abstract base class `BaseLPRouter` that defines the
common interface and shared functionality for LP-based routing approaches.
Concrete implementations include:
- `BipartiteAllocationRouter`: Per-qubit flow transitions
- `LpRouterUnified`: Per-operation-pair flow transitions

The base class handles:
- Coupling map and circuit setup
- Qubit operation tracking
- Operation layer construction
- Distance matrix computation
- Common LP constraints (operation uniqueness, position exclusivity)
- Solution extraction utilities

Subclasses must implement router-specific:
- Transition building
- Variable indexing for z variables
- Flow conservation constraints
- Objective function
- SWAP extraction

References
----------
- HiGHS Python Interface: https://ergo-code.github.io/HiGHS/dev/interfaces/python/
- Python ABC: https://docs.python.org/3/library/abc.html
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import highspy
import networkx as nx
import numpy as np

import quariadne.benchmarks.constants
import quariadne.circuit

if TYPE_CHECKING:
    import qiskit.transpiler

_logger = logging.getLogger(__name__)

# =============================================================================
# Type Aliases
# =============================================================================

type EdgeIndex = int
type OperationIndex = int
type TransitionIndex = int
type PhysicalEdge = tuple[
    quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit
]

# =============================================================================
# Constants
# =============================================================================

EQUALITY_BOUND_ONE = 1.0
EQUALITY_BOUND_ZERO = 0.0

COEFFICIENT_POSITIVE = 1.0
COEFFICIENT_NEGATIVE = -1.0

Y_VARIABLE_LOWER_BOUND = 0.0
Y_VARIABLE_UPPER_BOUND = 1.0
Z_VARIABLE_LOWER_BOUND = 0.0

CONSTRAINT_ARRAY_DTYPE = np.float64
INDEX_ARRAY_DTYPE = np.int32

VARIABLE_COUNT = "count"
VARIABLE_OFFSET = "offset"


# =============================================================================
# Shared Data Structures
# =============================================================================


@dataclass
class LPSolverOptions:
    """Configuration options for HiGHS LP solver.

    Attributes:
        output_flag: Whether to display solver output.
        log_to_console: Whether to log to console.
        solver: LP solver algorithm name.
        hipo_system_solver: System solver for HiPO.
        threads: Number of threads for parallel solving.

    Reference:
        - https://ergo-code.github.io/HiGHS/dev/options/definitions/
    """

    output_flag: bool = False
    log_to_console: bool = False
    solver: str = "simplex"  # HiPO fails on rank-deficient constraint matrices
    hipo_system_solver: str = quariadne.benchmarks.constants.HIPO_SYSTEM_SOLVER
    threads: int = quariadne.benchmarks.constants.HIPO_DEFAULT_THREADS


@dataclass
class QubitOperationInfo:
    """Tracks which operations involve each qubit and their ordering.

    Attributes:
        qubit: The logical qubit being tracked.
        operation_indices: List of operation indices involving this qubit (in order).
    """

    qubit: quariadne.circuit.LogicalQubit
    operation_indices: list[int] = field(default_factory=list)


# =============================================================================
# Cached Distance Computations
# =============================================================================


@functools.lru_cache(maxsize=8)
def compute_all_pairs_distances(
    edges: frozenset[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Compute shortest path distances for a coupling graph.

    Args:
        edges: Frozenset of (from_index, to_index) tuples.

    Returns:
        Dictionary mapping (source, target) to shortest path length.
    """
    graph: nx.Graph[int] = nx.Graph()
    for from_index, to_index in edges:
        graph.add_edge(from_index, to_index)

    distances: dict[tuple[int, int], int] = {}
    all_pairs = dict(nx.all_pairs_shortest_path_length(graph))

    for source, targets in all_pairs.items():
        for target, length in targets.items():
            distances[(source, target)] = length

    return distances


@functools.lru_cache(maxsize=8)
def compute_all_pairs_shortest_paths(
    edges: frozenset[tuple[int, int]],
) -> dict[tuple[int, int], list[int]]:
    """Compute shortest paths for all pairs of nodes.

    Args:
        edges: Frozenset of (from_index, to_index) tuples.

    Returns:
        Dictionary mapping (source, target) to list of node indices in path.
    """
    graph: nx.Graph[int] = nx.Graph()
    for from_index, to_index in edges:
        graph.add_edge(from_index, to_index)

    all_pairs_paths = dict(nx.all_pairs_shortest_path(graph))

    paths: dict[tuple[int, int], list[int]] = {}
    for source_node, target_paths in all_pairs_paths.items():
        for target_node, path_nodes in target_paths.items():
            paths[(source_node, target_node)] = path_nodes

    return paths


# =============================================================================
# Coupling Graph Conversion
# =============================================================================


def get_coupling_graph(
    coupling_map: qiskit.transpiler.CouplingMap,
) -> nx.DiGraph[quariadne.circuit.PhysicalQubit]:
    """Convert Qiskit coupling map to NetworkX DiGraph.

    Args:
        coupling_map: Qiskit CouplingMap object.

    Returns:
        NetworkX DiGraph with PhysicalQubit nodes.
    """
    coupling_graph_nx: nx.DiGraph[quariadne.circuit.PhysicalQubit] = nx.DiGraph()

    physical_qubits = tuple(
        quariadne.circuit.PhysicalQubit(q) for q in coupling_map.physical_qubits
    )
    physical_connections = tuple(
        (
            quariadne.circuit.PhysicalQubit(from_q),
            quariadne.circuit.PhysicalQubit(to_q),
        )
        for from_q, to_q in coupling_map.graph.edge_list()
    )

    coupling_graph_nx.add_nodes_from(physical_qubits)
    coupling_graph_nx.add_edges_from(physical_connections)

    return coupling_graph_nx


# =============================================================================
# Abstract Base Router Class
# =============================================================================


class BaseLPRouter(ABC):
    """Abstract base class for LP-based quantum circuit routers.

    Provides shared functionality for LP router implementations.
    Subclasses must implement abstract methods for router-specific logic.

    Attributes:
        coupling_map: NetworkX DiGraph representing hardware connectivity.
        routed_circuit: The quantum circuit to route.
        solver_options: Configuration for HiGHS solver.
        y_var_offset: Offset for y variables in the variable array.
        y_var_count: Total number of y variables.
    """

    # Type annotations for attributes set by subclasses in _setup_variable_indexing()
    y_var_offset: int
    y_var_count: int

    def __init__(
        self,
        coupling_map: nx.DiGraph[quariadne.circuit.PhysicalQubit],
        quantum_circuit: quariadne.circuit.AbstractQuantumCircuit,
        solver_options: LPSolverOptions | None = None,
        fixed_operation_edges: dict[int, int] | None = None,
    ) -> None:
        """Initialise the LP router using template method pattern.

        The initialisation follows these steps:
        1. Store basic attributes
        2. Call _preprocess_circuit() hook for circuit preparation
        3. Call _get_operations() hook to extract operations
        4. Build shared data structures
        5. Call _setup_variable_indexing() for router-specific indexing

        Args:
            coupling_map: NetworkX DiGraph representing hardware connectivity.
            quantum_circuit: The quantum circuit to route.
            solver_options: Configuration for HiGHS solver.
            fixed_operation_edges: Optional dict mapping operation index to
                fixed edge index.
        """
        self.coupling_map = coupling_map
        self.solver_options = solver_options or LPSolverOptions()
        self.fixed_operation_edges = fixed_operation_edges

        # Hook: Preprocess circuit (subclass can deepcopy, add dummy qubits, etc.)
        self.routed_circuit = self._preprocess_circuit(quantum_circuit)

        # Hook: Get operations to route (subclass can filter, etc.)
        self.operations = self._get_operations()
        self.operation_count = len(self.operations)

        # Extract edges from coupling map
        self.edges: list[PhysicalEdge] = list(coupling_map.edges())
        self.edge_count = len(self.edges)

        # Build edge lookup
        self.edge_to_index: dict[PhysicalEdge, int] = {
            edge: idx for idx, edge in enumerate(self.edges)
        }

        # Build qubit operation tracking
        self.qubit_operations = self._build_qubit_operation_map()

        # Build operation layers
        self.operation_layers = self._build_operation_layers()

        # Build distance matrix
        self.distance_matrix = self._build_distance_matrix()

    # -------------------------------------------------------------------------
    # Shared Methods
    # -------------------------------------------------------------------------

    def _build_qubit_operation_map(
        self,
    ) -> dict[quariadne.circuit.LogicalQubit, QubitOperationInfo]:
        """Build mapping from logical qubits to their participating operations."""
        qubit_ops: dict[quariadne.circuit.LogicalQubit, QubitOperationInfo] = {}

        for qubit in self.routed_circuit.qubits:
            qubit_ops[qubit] = QubitOperationInfo(qubit=qubit)

        for operation_index, operation in enumerate(self.operations):
            for qubit in operation.qubits_participating:
                qubit_ops[qubit].operation_indices.append(operation_index)

        return qubit_ops

    def _build_operation_layers(self) -> list[list[int]]:
        """Group operations into parallel layers based on qubit dependencies."""
        if self.operation_count == 0:
            return []

        layers: list[list[int]] = []
        qubit_last_layer: dict[quariadne.circuit.LogicalQubit, int] = {}

        for operation_index, operation in enumerate(self.operations):
            qubits = operation.qubits_participating

            min_layer = 0
            for qubit in qubits:
                if qubit in qubit_last_layer:
                    min_layer = max(min_layer, qubit_last_layer[qubit] + 1)

            while len(layers) <= min_layer:
                layers.append([])

            layers[min_layer].append(operation_index)

            for qubit in qubits:
                qubit_last_layer[qubit] = min_layer

        return layers

    def _build_distance_matrix(
        self,
    ) -> dict[
        tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit],
        float,
    ]:
        """Build distance matrix for physical qubit pairs."""
        coupling_edges = frozenset(
            (edge[0].index, edge[1].index) for edge in self.coupling_map.edges()
        )
        index_distances = compute_all_pairs_distances(coupling_edges)

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

    def _get_qubit_position_in_edge(
        self,
        qubit: quariadne.circuit.LogicalQubit,
        operation_index: int,
        edge: PhysicalEdge,
    ) -> quariadne.circuit.PhysicalQubit:
        """Determine which physical position a qubit occupies in an edge assignment.

        For operation (A, B) on edge (u, v), qubit A sits at u and qubit B at v.
        """
        operation = self.operations[operation_index]
        left_qubit, right_qubit = operation.qubits_participating
        left_physical, right_physical = edge

        if qubit == left_qubit:
            return left_physical
        if qubit == right_qubit:
            return right_physical

        raise ValueError(
            f"Qubit {qubit} is not involved in operation {operation_index}"
        )

    def _get_y_var_index(self, operation_index: int, edge_index: int) -> int:
        """Get the variable index for y_{operation, edge}."""
        return self.y_var_offset + operation_index * self.edge_count + edge_index

    def _add_operation_uniqueness_constraints(self, highs_model: highspy.Highs) -> None:
        """Add constraints: each operation assigned to exactly one edge (C1)."""
        num_constraints = self.operation_count
        nonzeros_per_constraint = self.edge_count
        total_nonzeros = num_constraints * nonzeros_per_constraint

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

    def _add_position_exclusivity_constraints(self, highs_model: highspy.Highs) -> None:
        """Add constraints: each position used at most once per layer (C2)."""
        position_to_edge_indices: dict[quariadne.circuit.PhysicalQubit, list[int]] = {
            node: [] for node in self.coupling_map.nodes
        }

        for edge_index, edge in enumerate(self.edges):
            source, target = edge
            position_to_edge_indices[source].append(edge_index)
            position_to_edge_indices[target].append(edge_index)

        multi_operation_layers = [
            layer for layer in self.operation_layers if len(layer) > 1
        ]

        for layer in multi_operation_layers:
            for position, edge_indices in position_to_edge_indices.items():
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

    def _add_fixed_operation_constraints(self, highs_model: highspy.Highs) -> None:
        """Fix y variables for operations with pre-assigned edges."""
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

    def _extract_operation_edges(
        self, col_values: list[float]
    ) -> dict[int, PhysicalEdge]:
        """Extract operation-to-edge assignments from y variable values."""
        y_values = np.array(col_values[: self.y_var_count]).reshape(
            self.operation_count, self.edge_count
        )
        best_edge_indices = np.argmax(y_values, axis=1)

        return {
            operation_index: self.edges[best_edge_indices[operation_index]]
            for operation_index in range(self.operation_count)
        }

    def _extract_initial_mapping(
        self, operation_edge_assignment: dict[int, PhysicalEdge]
    ) -> dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]:
        """Extract initial qubit mapping from first operation assignments."""
        initial_mapping: dict[
            quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit
        ] = {}

        for qubit, op_info in self.qubit_operations.items():
            if op_info.operation_indices:
                first_op_index = op_info.operation_indices[0]
                edge = operation_edge_assignment[first_op_index]
                position = self._get_qubit_position_in_edge(qubit, first_op_index, edge)
                initial_mapping[qubit] = position

        return initial_mapping

    def _extract_final_mapping(
        self, operation_edge_assignment: dict[int, PhysicalEdge]
    ) -> dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]:
        """Extract final qubit mapping from last operation assignments."""
        final_mapping: dict[
            quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit
        ] = {}

        for qubit, op_info in self.qubit_operations.items():
            if op_info.operation_indices:
                last_op_index = op_info.operation_indices[-1]
                edge = operation_edge_assignment[last_op_index]
                position = self._get_qubit_position_in_edge(qubit, last_op_index, edge)
                final_mapping[qubit] = position

        return final_mapping

    def _build_operation_to_edge_by_qubits(
        self, operation_edge_assignment: dict[int, PhysicalEdge]
    ) -> dict[tuple[int, int, int], PhysicalEdge]:
        """Build lookup from (qubit1, qubit2, occurrence) to edge."""
        qubit_pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        result: dict[tuple[int, int, int], PhysicalEdge] = {}

        for operation_index, edge in operation_edge_assignment.items():
            operation = self.operations[operation_index]
            q1, q2 = operation.qubits_participating
            pair_key = (q1.index, q2.index)
            occurrence = qubit_pair_counts[pair_key]
            qubit_pair_counts[pair_key] += 1
            result[(q1.index, q2.index, occurrence)] = edge

        return result

    # -------------------------------------------------------------------------
    # Abstract Methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    def _preprocess_circuit(
        self, quantum_circuit: quariadne.circuit.AbstractQuantumCircuit
    ) -> quariadne.circuit.AbstractQuantumCircuit:
        """Preprocess the circuit before routing.

        Hook for subclasses to deepcopy, add dummy qubits, etc.

        Args:
            quantum_circuit: The input quantum circuit.

        Returns:
            The preprocessed circuit to route.
        """
        ...

    @abstractmethod
    def _get_operations(self) -> list[quariadne.circuit.QuantumOperation]:
        """Get the list of operations to route.

        Hook for subclasses to filter operations (e.g., only two-qubit gates).

        Returns:
            List of operations to include in routing.
        """
        ...

    @abstractmethod
    def _setup_variable_indexing(self) -> None:
        """Set up variable indexing for the LP formulation."""
        ...

    @abstractmethod
    def _add_flow_conservation_constraints(self, highs_model: highspy.Highs) -> None:
        """Add flow conservation constraints (C3)."""
        ...

    @abstractmethod
    def _set_objective(self, highs_model: highspy.Highs) -> None:
        """Set the objective function."""
        ...

    @abstractmethod
    def run(self):
        """Solve the LP and return the routing result."""
        ...
