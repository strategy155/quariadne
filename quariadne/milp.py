import typing
import networkx as nx
import numpy as np
from dataclasses import dataclass, field, asdict
from enum import Enum

import quariadne.circuit
import scipy.optimize
import scipy.sparse
import qiskit.transpiler

if typing.TYPE_CHECKING:
    from quariadne.routers import OperationEdgeAssignment

DEFAULT_SHAPE_TYPE = np.int64
DEFAULT_CONSTRAINT_TYPE = np.float64

# Optimisation coefficient constants
QUBIT_MOVEMENT_PENALTY_COEFFICIENT = 0.5
MILP_SOLVER_TIMEOUT_SECONDS = 3600

# Constraint name constants
LOGICAL_UNIQUENESS_CONSTRAINT = "logical_uniqueness_constraint"
PHYSICAL_UNIQUENESS_CONSTRAINT = "physical_uniqueness_constraint"
GATE_EXECUTION_CONSTRAINT = "gate_execution_constraint"
GATE_MAPPING_CONSTRAINT = "gate_mapping_constraint"
GATE_MAPPING_LEFT_QUBIT_CONSTRAINT = "gate_mapping_left_qubit_constraint"
GATE_MAPPING_RIGHT_QUBIT_CONSTRAINT = "gate_mapping_right_qubit_constraint"
GATE_MAPPING_FULL_QUBIT_CONSTRAINT = "gate_mapping_full_qubit_constraint"
FLOW_CONDITION_IN_CONSTRAINT = "flow_condition_in_constraint"
FLOW_CONDITION_OUT_CONSTRAINT = "flow_condition_out_constraint"
GATE_EXECUTION_SWAP_CONSTRAINT = "gate_execution_swap_constraint"
NON_GATE_SWAP_CONSTRAINT = "non_gate_swap_constraint"
FIXED_MAPPING_CONSTRAINT = "fixed_mapping_constraint"
FIXED_EDGES_CONSTRAINT = "fixed_edges_constraint"
GATE_EXECUTION_LAYER_EDGE_UNIQUENESS_CONSTRAINT = (
    "gate_execution_layer_edge_uniqueness_constraint"
)

# Constraint bound value constants
ONE_EQUALITY_CONSTRAINT_BOUND = 1
ZERO_EQUALITY_CONSTRAINT_BOUND = 0
OPEN_LOWER_CONSTRAINT_BOUND = -np.inf
OPEN_UPPER_CONSTRAINT_BOUND = np.inf
MINUS_ONE_LOWER_BOUND = -1

# Variable bound constants
BINARY_VARIABLE_LOWER_BOUND = 0
BINARY_VARIABLE_UPPER_BOUND = 1
INTEGER_VARIABLE_INTEGRALITY = 1
CONTINUOUS_VARIABLE_INTEGRALITY = 0

type VariableShape = tuple[int, ...]
type SwapEdge = tuple[int, int]
type GateLayer = tuple[quariadne.circuit.QuantumOperation, ...]


class RoutingVariableType(Enum):
    """Enum for variable types in MILP optimization."""

    MAPPING = "mapping"
    GATE_EXECUTION = "gate_execution"
    QUBIT_MOVEMENT = "qubit_movement"


# Variable registry field constants
VARIABLE_SHAPE = "shape"
VARIABLE_FLAT_SHAPE = "flat_shape"
VARIABLE_OFFSET = "offset"


@dataclass
class ConstraintMatrixSparseData:
    """Container for sparse constraint matrix data during MILP constraint generation.

    Holds the coefficient data, row indices, and column indices needed to construct
    sparse constraint coefficient matrices efficiently. Shape is inferred during creation.

    Attributes:
        data: List of non-zero coefficient values
        row_indices: List of row indices corresponding to each coefficient
        col_indices: List of column indices corresponding to each coefficient
    """

    data: list[float] = field(default_factory=list)
    row_indices: list[int] = field(default_factory=list)
    col_indices: list[int] = field(default_factory=list)

    def add_coefficient(
        self, value: float, row_index: int, col_index: int
    ) -> None:  # TODO: Fix typing!
        """Add a coefficient triplet to the sparse matrix data.

        Args:
            value: Coefficient value to add
            row_index: Row index for this coefficient
            col_index: Column index for this coefficient
        """
        self.data.append(value)
        self.row_indices.append(row_index)
        self.col_indices.append(col_index)


@dataclass
class MilpRouterResult:
    """Result container for MILP router optimization with processed variable matrices.

    Contains the three groups of decision variables from the MILP solution in unflattened form:
    - mapping_variables: qubit-to-physical mapping at each timestep
    - gate_execution_variables: which gates execute on which physical edges
    - qubit_movement_variables: qubit movement between physical locations

    This is a pure data container. Use IlpRouter for processing and extracting routing information.
    """

    milp_result: scipy.optimize.OptimizeResult
    mapping_variables: np.ndarray
    gate_execution_variables: np.ndarray
    qubit_movement_variables: np.ndarray
    worst_spacing: int


@dataclass
class HiGHSSolverOptions:
    """Configuration options for the HiGHS MILP solver.

    Attributes:
        time_limit: Maximum solver runtime in seconds. If the solver exceeds
                   this limit, it returns the best solution found so far.
    """

    time_limit: float
    write_model_to_file: bool


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

    Example:
        >>> from qiskit_ibm_runtime.fake_provider import FakeManilaV2
        >>> backend = FakeManilaV2()
        >>> coupling_graph = get_coupling_graph(backend.coupling_map)
        >>> print(coupling_graph.nodes())
        >>> print(coupling_graph.edges())
    """
    coupling_graph_nx: nx.DiGraph[quariadne.circuit.PhysicalQubit] = nx.DiGraph()

    # Extract physical qubits from coupling map
    coupling_map_qubits = coupling_map.physical_qubits
    physical_qubits = tuple(
        quariadne.circuit.PhysicalQubit(coupling_map_qubit)
        for coupling_map_qubit in coupling_map_qubits
    )

    # Extract edges from coupling map
    coupling_map_edgelist = coupling_map.graph.edge_list()
    physical_qubits_connections = tuple(
        (
            quariadne.circuit.PhysicalQubit(coupling_map_from),
            quariadne.circuit.PhysicalQubit(coupling_map_to),
        )
        for coupling_map_from, coupling_map_to in coupling_map_edgelist
    )

    # Build NetworkX graph
    coupling_graph_nx.add_nodes_from(physical_qubits)
    coupling_graph_nx.add_edges_from(physical_qubits_connections)

    return coupling_graph_nx


def create_gate_layers(
    operations: list[quariadne.circuit.QuantumOperation],
    coupling_map: nx.Graph,
) -> list[GateLayer]:
    """Create gate layers by grouping operations with non-conflicting qubits.

    Groups two-qubit operations into layers where operations in the same layer
    can execute in parallel (no qubit conflicts). Uses a greedy algorithm that
    iterates through operations in order and adds each operation to the current
    layer if no qubit conflict exists, otherwise starts a new layer.

    Additionally enforces that the layer size does not exceed the maximum
    matching size of the coupling map, which represents the maximum number of
    two-qubit gates that can execute simultaneously on the hardware.

    Args:
        operations: List of two-qubit operations to be grouped into layers
        coupling_map: Physical qubit connectivity graph (directed or undirected)

    Returns:
        List of gate layers, where each layer contains operations that can
        execute in parallel.
    """
    # Calculate maximum matching size for the coupling map
    # This represents the maximum number of two-qubit gates that can execute
    # simultaneously on the hardware
    coupling_map_undirected = coupling_map.to_undirected()
    maximum_matching = nx.algorithms.matching.maximal_matching(coupling_map_undirected)
    max_matching_size = len(maximum_matching)

    gate_layers: list[GateLayer] = []
    current_layer: list[quariadne.circuit.QuantumOperation] = []
    current_layer_qubits: set[quariadne.circuit.LogicalQubit] = set()

    for operation in operations:
        operation_qubits = set(operation.qubits_participating)

        # Check if operation qubits conflict with current layer qubits, or
        # if adding this operation would exceed the maximum matching constraint
        qubit_conflict_exists = bool(current_layer_qubits & operation_qubits)
        exceeds_max_matching = len(current_layer) >= max_matching_size

        if qubit_conflict_exists or exceeds_max_matching:
            # Conflict exists or max matching exceeded, start new layer
            # Convert list to immutable tuple for hashable GateLayer type
            completed_layer: GateLayer = tuple(current_layer)
            gate_layers.append(completed_layer)
            current_layer = [operation]
            current_layer_qubits = operation_qubits
        else:
            # No conflict and within max matching limit, add to current layer
            current_layer.append(operation)
            current_layer_qubits.update(operation_qubits)

    # Add the last layer if not empty
    if current_layer:
        # Convert list to immutable tuple for hashable GateLayer type
        final_layer: GateLayer = tuple(current_layer)
        gate_layers.append(final_layer)

    return gate_layers


class MilpScipyRouter:
    """Mixed-Integer Linear Programming router using scipy.optimize.milp for quantum circuit qubit routing.

    Implements the MILP formulation and constraint generation for the quantum circuit routing problem.
    This class handles the mathematical optimisation using scipy but does not process the results.
    Use IlpRouter for complete routing functionality including result processing.

    The MILP formulation includes three types of decision variables:
    - Mapping variables: qubit-to-physical mapping at each timestep
    - Gate execution variables: which gates execute on which physical edges
    - Qubit movement variables: qubit movement between physical locations

    Attributes:
        coupling_map: NetworkX DiGraph representing hardware connectivity
        qubits: Tuple of logical qubits (including dummy qubits)
        qubit_count: Number of physical qubits available
        operations: List of two-qubit operations to be routed
        operation_count: Number of operations to route
        worst_spacing: Worst-case timesteps for token swapping
        spaced_timesteps_count: Total timesteps in routing schedule
    """

    def _add_dummy_qubits(
        self, qubits: tuple[quariadne.circuit.LogicalQubit, ...]
    ) -> tuple[quariadne.circuit.LogicalQubit, ...]:
        """Add dummy logical qubits to match hardware qubit count.

        Ensures the number of logical qubits matches the number of physical qubits
        available on the hardware. This is required for the MILP formulation to work
        correctly as it assumes a bijective mapping between logical and physical qubits.

        Args:
            qubits: Tuple of logical qubits to extend with dummy qubits

        Returns:
            Extended tuple of logical qubits including dummy qubits

        Raises:
            TypeError: If more logical qubits than available physical qubits.
        """
        qubit_count = len(qubits)
        if qubit_count < self.coupling_map.number_of_nodes():
            dummy_logical_qubits = tuple(
                quariadne.circuit.LogicalQubit(dummy_index)
                for dummy_index in range(qubit_count, self.qubit_count)
            )
            return qubits + dummy_logical_qubits

        elif qubit_count > self.coupling_map.number_of_nodes():
            raise TypeError("We got more qubits than we can route!")

        return qubits

    def __init__(
        self,
        coupling_map: nx.DiGraph,
        gate_layers: list[GateLayer],
        qubits: tuple[quariadne.circuit.LogicalQubit, ...],
        integrality=INTEGER_VARIABLE_INTEGRALITY,
        fixed_mapping: np.ndarray | None = None,
        fixed_edges: list[list["OperationEdgeAssignment"]] | None = None,
    ):
        """Initialize the MILP router with hardware topology, gate layers, and qubits.

        Sets up all necessary data structures for the MILP formulation including:
        - Variable shapes and offsets for mapping, gate execution, and qubit movement
        - Constraint generation registry with bounds
        - Timestep calculations based on worst-case token swapping

        Args:
            coupling_map: NetworkX DiGraph representing the physical qubit connectivity
                         and allowed two-qubit gate operations on the hardware.
            gate_layers: Pre-computed gate layers where operations in each layer can
                        execute in parallel (no qubit conflicts).
            qubits: Tuple of logical qubits to be mapped to physical qubits.
            integrality: Integrality constraint value for decision variables. Use
                        INTEGER_VARIABLE_INTEGRALITY (1) for MILP or 0 for LP relaxation.
            fixed_mapping: Optional numpy array specifying fixed mapping constraints.
                          Should have shape matching the mapping variables format.
            fixed_edges: Optional list of edge assignment lists per gate layer. Each element
                        is a list of (operation_index, edge_index) tuples for that layer.
                        Position i in the list corresponds to gate_layers[i].
        """

        self.integrality = integrality
        self.fixed_mapping = fixed_mapping
        self.fixed_edges = fixed_edges
        self.coupling_map = coupling_map
        self.qubit_count = coupling_map.number_of_nodes()

        # Store gate layers and derive operations list
        self.gate_layers = gate_layers
        self.layer_count = len(self.gate_layers)

        # Flatten layers to get operations list
        self.operations = [
            operation for layer in self.gate_layers for operation in layer
        ]
        self.operation_count = len(self.operations)

        # Calculate worst spacing before adding dummy qubits
        # worst spacing is the token swapping worst case n^2
        self.worst_spacing = len(qubits)

        # Add dummy qubits to match hardware qubit count
        self.qubits = self._add_dummy_qubits(qubits)

        self.spaced_timesteps_count = self.layer_count * self.worst_spacing

        # calculating the shapes of the decision variables
        self.qubit_movement_shape: VariableShape = (
            self.spaced_timesteps_count,
            self.qubit_count,
            self.qubit_count,
            self.qubit_count,
        )
        self.mapping_variables_shape: VariableShape = (
            self.spaced_timesteps_count,
            self.qubit_count,
            self.qubit_count,
        )
        self.gate_execution_variables_shape: VariableShape = (
            self.operation_count,
            self.coupling_map.number_of_edges(),
        )

        self.flat_qubit_movement_shape: np.integer = np.prod(
            self.qubit_movement_shape, dtype=DEFAULT_SHAPE_TYPE
        )
        self.flat_mapping_variables_shape: np.integer = np.prod(
            self.mapping_variables_shape, dtype=DEFAULT_SHAPE_TYPE
        )
        self.flat_gate_execution_variables_shape: np.integer = np.prod(
            self.gate_execution_variables_shape, dtype=DEFAULT_SHAPE_TYPE
        )
        self.full_decision_variables_shape: np.integer = (
            self.flat_mapping_variables_shape
            + self.flat_gate_execution_variables_shape
            + self.flat_qubit_movement_shape
        )

        # Precalculated variable offsets
        self.mapping_variables_offset: np.integer = DEFAULT_SHAPE_TYPE(0)
        self.gate_execution_variables_offset: np.integer = (
            self.flat_mapping_variables_shape
        )
        self.qubit_movement_variables_offset: np.integer = (
            self.flat_mapping_variables_shape + self.flat_gate_execution_variables_shape
        )

        self.coupling_map_edges = list(self.coupling_map.edges())

        # Variable type registry with shapes, flat shapes, and offsets
        self.variable_types = {
            RoutingVariableType.MAPPING: {
                VARIABLE_SHAPE: self.mapping_variables_shape,
                VARIABLE_FLAT_SHAPE: self.flat_mapping_variables_shape,
                VARIABLE_OFFSET: self.mapping_variables_offset,
            },
            RoutingVariableType.GATE_EXECUTION: {
                VARIABLE_SHAPE: self.gate_execution_variables_shape,
                VARIABLE_FLAT_SHAPE: self.flat_gate_execution_variables_shape,
                VARIABLE_OFFSET: self.gate_execution_variables_offset,
            },
            RoutingVariableType.QUBIT_MOVEMENT: {
                VARIABLE_SHAPE: self.qubit_movement_shape,
                VARIABLE_FLAT_SHAPE: self.flat_qubit_movement_shape,
                VARIABLE_OFFSET: self.qubit_movement_variables_offset,
            },
        }

        # Constraint generator methods registry
        self.constraint_generators = {
            LOGICAL_UNIQUENESS_CONSTRAINT: self._generate_logical_uniqueness_constraint,
            PHYSICAL_UNIQUENESS_CONSTRAINT: self._generate_physical_uniqueness_constraint,
            GATE_EXECUTION_CONSTRAINT: self._generate_gate_execution_constraint,
            GATE_EXECUTION_LAYER_EDGE_UNIQUENESS_CONSTRAINT: self._generate_gate_execution_layer_edge_uniqueness_constraint,
            GATE_MAPPING_CONSTRAINT: self._generate_gate_mapping_constraint,
            GATE_MAPPING_LEFT_QUBIT_CONSTRAINT: self._generate_gate_mapping_left_qubit_constraint,
            GATE_MAPPING_RIGHT_QUBIT_CONSTRAINT: self._generate_gate_mapping_right_qubit_constraint,
            GATE_MAPPING_FULL_QUBIT_CONSTRAINT: self._generate_gate_mapping_full_qubit_constraint,
            FLOW_CONDITION_IN_CONSTRAINT: self._generate_flow_condition_in_constraint,
            FLOW_CONDITION_OUT_CONSTRAINT: self._generate_flow_condition_out_constraint,
            GATE_EXECUTION_SWAP_CONSTRAINT: self._generate_gate_execution_swap_constraint,
            NON_GATE_SWAP_CONSTRAINT: self._generate_non_gate_swap_constraint,
            FIXED_MAPPING_CONSTRAINT: self._generate_fixed_mapping_constraint,
            FIXED_EDGES_CONSTRAINT: self._generate_fixed_edges_constraint,
        }

        # Constraint lower bounds registry
        self.constraint_lower_bounds = {
            LOGICAL_UNIQUENESS_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            PHYSICAL_UNIQUENESS_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            GATE_EXECUTION_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            GATE_EXECUTION_LAYER_EDGE_UNIQUENESS_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            GATE_MAPPING_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            GATE_MAPPING_LEFT_QUBIT_CONSTRAINT: OPEN_LOWER_CONSTRAINT_BOUND,
            GATE_MAPPING_RIGHT_QUBIT_CONSTRAINT: OPEN_LOWER_CONSTRAINT_BOUND,
            GATE_MAPPING_FULL_QUBIT_CONSTRAINT: MINUS_ONE_LOWER_BOUND,
            FLOW_CONDITION_IN_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            FLOW_CONDITION_OUT_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            GATE_EXECUTION_SWAP_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            NON_GATE_SWAP_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            FIXED_MAPPING_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            FIXED_EDGES_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
        }

        # Constraint upper bounds registry
        self.constraint_upper_bounds = {
            LOGICAL_UNIQUENESS_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            PHYSICAL_UNIQUENESS_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            GATE_EXECUTION_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            GATE_EXECUTION_LAYER_EDGE_UNIQUENESS_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            GATE_MAPPING_CONSTRAINT: OPEN_UPPER_CONSTRAINT_BOUND,
            GATE_MAPPING_LEFT_QUBIT_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            GATE_MAPPING_RIGHT_QUBIT_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            GATE_MAPPING_FULL_QUBIT_CONSTRAINT: OPEN_UPPER_CONSTRAINT_BOUND,
            FLOW_CONDITION_IN_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            FLOW_CONDITION_OUT_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            GATE_EXECUTION_SWAP_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            NON_GATE_SWAP_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            FIXED_MAPPING_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            FIXED_EDGES_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
        }

    # TODO: CHECK THE UPDATES OF THE CSR
    @typing.no_type_check
    def _build_sparse_coefficient_matrix(
        self, sparse_data: ConstraintMatrixSparseData, constraint_count: int
    ) -> scipy.sparse.csr_array:
        """Build a sparse coefficient matrix from constraint matrix sparse data.

        Args:
            sparse_data: ConstraintMatrixSparseData containing coefficient values and indices

        Returns:
            Sparse coefficient matrix in CSR format for efficient operations.
        """
        coefficient_matrix = scipy.sparse.csr_array(
            (sparse_data.data, (sparse_data.row_indices, sparse_data.col_indices)),
            shape=(constraint_count, self.full_decision_variables_shape),
            dtype=DEFAULT_CONSTRAINT_TYPE,
        )
        return coefficient_matrix

    def _generate_logical_uniqueness_constraint(self):
        """Generate logical qubit uniqueness constraint coefficients.

        Ensures each logical qubit maps to exactly one physical qubit at each timestep.

        Returns:
            Coefficient matrix for logical qubit uniqueness constraints.
        """
        # Create sparse data container with default empty lists
        logical_uniqueness_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        # Iterate through all timesteps in the routing schedule
        for timestep in range(self.spaced_timesteps_count):
            # For each logical qubit, ensure it maps to exactly one physical qubit
            for logical_qubit in self.qubits:
                for physical_qubit in self.coupling_map.nodes:
                    # Build multidimensional index for mapping variable
                    logical_by_physical_index = (
                        timestep,
                        physical_qubit.index,
                        logical_qubit.index,
                    )
                    # Convert to flat index for coefficient matrix
                    flattened_logical_by_physical_index = np.ravel_multi_index(
                        logical_by_physical_index, self.mapping_variables_shape
                    )
                    # Add coefficient of 1 for this mapping variable
                    logical_uniqueness_sparse_data.add_coefficient(
                        1.0, constraint_index, flattened_logical_by_physical_index
                    )

                # Move to next constraint row after processing all physical qubits for this logical qubit
                constraint_index += 1

        # Build sparse coefficient matrix
        # Total constraints = final constraint_index (number of constraints processed)
        logical_uniqueness_constraint = self._build_sparse_coefficient_matrix(
            logical_uniqueness_sparse_data, constraint_index
        )
        return logical_uniqueness_constraint

    def _generate_physical_uniqueness_constraint(self):
        """Generate physical qubit uniqueness constraint coefficients.

        Ensures each physical qubit hosts exactly one logical qubit at each timestep.

        Returns:
            Coefficient matrix for physical qubit uniqueness constraints.
        """
        # Create sparse data container with default empty lists
        physical_uniqueness_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        # Iterate through all timesteps in the routing schedule
        for timestep in range(self.spaced_timesteps_count):
            # For each physical qubit, ensure it hosts exactly one logical qubit
            for physical_qubit in self.coupling_map.nodes:
                for logical_qubit in self.qubits:
                    # Build multidimensional index for mapping variable
                    physical_by_logical_index = (
                        timestep,
                        physical_qubit.index,
                        logical_qubit.index,
                    )
                    # Convert to flat index for coefficient matrix
                    flattened_physical_by_logical_index = np.ravel_multi_index(
                        physical_by_logical_index, self.mapping_variables_shape
                    )
                    # Add coefficient of 1 for this mapping variable
                    physical_uniqueness_sparse_data.add_coefficient(
                        1.0, constraint_index, flattened_physical_by_logical_index
                    )

                # Move to next constraint row after processing all logical qubits for this physical qubit
                constraint_index += 1

        # Build sparse coefficient matrix
        # Total constraints = final constraint_index (number of constraints processed)
        physical_uniqueness_constraint = self._build_sparse_coefficient_matrix(
            physical_uniqueness_sparse_data, constraint_index
        )
        return physical_uniqueness_constraint

    def _generate_gate_execution_constraint(self):
        """Generate gate execution uniqueness constraint coefficients.

        Ensures each gate executes on exactly one physical edge at its assigned timestep.
        With gate layering, multiple gates in the same layer execute at the same timestep.

        Returns:
            Coefficient matrix for gate execution uniqueness constraints.
        """
        # Create sparse data container with default empty lists
        gate_execution_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        # For each layer, process all operations that execute in parallel
        for layer_index in range(self.layer_count):
            # For each operation in this layer
            for operation in self.gate_layers[layer_index]:
                # Get the operation index for variable indexing
                operation_index = self.operations.index(operation)

                # Consider all possible physical edges where the gate could execute
                for physical_edge_index in range(self.coupling_map.number_of_edges()):
                    # Build multidimensional index for gate execution variable (2D)
                    gate_execution_multi_index = (
                        operation_index,
                        physical_edge_index,
                    )
                    # Convert to flat index within gate execution variable space
                    gate_execution_ravel_index = np.ravel_multi_index(
                        gate_execution_multi_index, self.gate_execution_variables_shape
                    )
                    # Apply offset to position correctly in full decision variable vector
                    flattened_gate_execution_index = (
                        self.flat_mapping_variables_shape + gate_execution_ravel_index
                    )
                    # Add coefficient of 1 for this gate execution variable
                    gate_execution_sparse_data.add_coefficient(
                        1.0, constraint_index, flattened_gate_execution_index
                    )

                # Move to next constraint row after processing all physical edges for this operation
                constraint_index += 1

        # Build sparse coefficient matrix
        # Total constraints = final constraint_index (number of constraints processed)
        gate_execution_constraint = self._build_sparse_coefficient_matrix(
            gate_execution_sparse_data, constraint_index
        )
        return gate_execution_constraint

    def _generate_gate_execution_layer_edge_uniqueness_constraint(self):
        """Generate gate execution layer-edge uniqueness constraint coefficients.

        Ensures at most one operation from each layer executes on any given physical edge.
        This prevents multiple gates in the same layer from conflicting on the same edge,
        which is essential for parallel gate execution.

        Returns:
            Coefficient matrix for gate execution layer-edge uniqueness constraints.
        """
        # Create sparse data container with default empty lists
        layer_edge_uniqueness_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        # For each layer
        for layer_index in range(self.layer_count):
            # For each physical edge in the coupling map
            for physical_edge_index in range(self.coupling_map.number_of_edges()):
                # Sum over all operations in this layer for this edge
                for operation in self.gate_layers[layer_index]:
                    # Get the operation index for variable indexing
                    operation_index = self.operations.index(operation)

                    # Build multidimensional index for gate execution variable (2D)
                    gate_execution_multi_index = (
                        operation_index,
                        physical_edge_index,
                    )
                    # Convert to flat index within gate execution variable space
                    gate_execution_ravel_index = np.ravel_multi_index(
                        gate_execution_multi_index, self.gate_execution_variables_shape
                    )
                    # Apply offset to position correctly in full decision variable vector
                    flattened_gate_execution_index = (
                        self.flat_mapping_variables_shape + gate_execution_ravel_index
                    )
                    # Add coefficient of 1 for this gate execution variable
                    layer_edge_uniqueness_sparse_data.add_coefficient(
                        1.0, constraint_index, flattened_gate_execution_index
                    )

                # Move to next constraint row after processing all operations for this layer-edge combination
                constraint_index += 1

        # Build sparse coefficient matrix
        # Total constraints = layer_count × edge_count
        layer_edge_uniqueness_constraint = self._build_sparse_coefficient_matrix(
            layer_edge_uniqueness_sparse_data, constraint_index
        )
        return layer_edge_uniqueness_constraint

    def _generate_gate_mapping_constraint(self):
        """Generate gate mapping constraint coefficients using McCormick relaxation.

        Enforces the basic constraint for gate execution variables.
        This is part of the McCormick relaxation for gate mapping constraints.

        Returns:
            Coefficient matrix for gate mapping constraints.
        """
        # Create sparse data container with default empty lists
        gate_mapping_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        # For each layer, process all operations that execute in parallel
        for layer_index in range(self.layer_count):
            # For each operation in this layer
            for operation in self.gate_layers[layer_index]:
                # Get the operation index for variable indexing
                operation_index = self.operations.index(operation)

                for physical_edge_index in range(self.coupling_map.number_of_edges()):
                    # Build multidimensional index for gate execution variable (2D)
                    gate_execution_multi_index = (
                        operation_index,
                        physical_edge_index,
                    )
                    # Convert to flat index within gate execution variable space
                    gate_execution_ravel_index = np.ravel_multi_index(
                        gate_execution_multi_index, self.gate_execution_variables_shape
                    )
                    # Apply offset to position correctly in full decision variable vector
                    flattened_gate_execution_index = (
                        self.flat_mapping_variables_shape + gate_execution_ravel_index
                    )
                    # Add coefficient of 1 for gate execution variable
                    gate_mapping_sparse_data.add_coefficient(
                        1.0, constraint_index, flattened_gate_execution_index
                    )

                    # Move to next constraint row after processing this operation-edge combination
                    constraint_index += 1

        # Build sparse coefficient matrix
        # Total constraints = final constraint_index (number of constraints processed)
        gate_mapping_constraint = self._build_sparse_coefficient_matrix(
            gate_mapping_sparse_data, constraint_index
        )
        return gate_mapping_constraint

    def _generate_gate_mapping_left_qubit_constraint(self):
        """Generate left qubit mapping constraint coefficients using McCormick relaxation.

        Enforces the constraint between gate execution and left qubit mapping variables.
        This is part of the McCormick relaxation for gate mapping constraints.

        Returns:
            Coefficient matrix for left qubit mapping constraints.
        """
        # Create sparse data container with default empty lists
        gate_mapping_left_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        # For each layer, process all operations that execute in parallel
        for layer_index in range(self.layer_count):
            # Calculate the timestep when operations in this layer execute
            spaced_timestep = layer_index * self.worst_spacing

            # For each operation in this layer
            for operation in self.gate_layers[layer_index]:
                # Get the operation index for variable indexing
                operation_index = self.operations.index(operation)

                # Get the logical qubits participating in this operation
                left_logical_qubit, right_logical_qubit = operation.qubits_participating

                for physical_edge_index in range(self.coupling_map.number_of_edges()):
                    # Build multidimensional index for gate execution variable (2D)
                    gate_execution_multi_index = (
                        operation_index,
                        physical_edge_index,
                    )
                    # Convert to flat index within gate execution variable space
                    gate_execution_ravel_index = np.ravel_multi_index(
                        gate_execution_multi_index, self.gate_execution_variables_shape
                    )
                    # Apply offset to position correctly in full decision variable vector
                    flattened_gate_execution_index = (
                        self.flat_mapping_variables_shape + gate_execution_ravel_index
                    )
                    # Add coefficient of 1 for gate execution variable
                    gate_mapping_left_sparse_data.add_coefficient(
                        1.0, constraint_index, flattened_gate_execution_index
                    )

                    # Get the physical edge (left and right physical qubits)
                    physical_edge = self.coupling_map_edges[physical_edge_index]
                    left_physical_qubit, right_physical_qubit = physical_edge

                    # Build multidimensional index for left qubit mapping variable
                    left_qubit_mapping_multi_index = (
                        spaced_timestep,
                        left_physical_qubit.index,
                        left_logical_qubit.index,
                    )
                    # Convert to flat index for left qubit mapping variable
                    left_qubit_mapping_ravel_index = np.ravel_multi_index(
                        left_qubit_mapping_multi_index, self.mapping_variables_shape
                    )
                    # Add coefficient of -1 for left qubit mapping variable
                    gate_mapping_left_sparse_data.add_coefficient(
                        -1.0, constraint_index, left_qubit_mapping_ravel_index
                    )

                    # Move to next constraint row after processing this operation-edge combination
                    constraint_index += 1

        # Build sparse coefficient matrix
        # Total constraints = final constraint_index (number of constraints processed)
        gate_mapping_left_constraint = self._build_sparse_coefficient_matrix(
            gate_mapping_left_sparse_data, constraint_index
        )
        return gate_mapping_left_constraint

    def _generate_gate_mapping_right_qubit_constraint(self):
        """Generate right qubit mapping constraint coefficients using McCormick relaxation.

        Enforces the constraint between gate execution and right qubit mapping variables.
        This is part of the McCormick relaxation for gate mapping constraints.

        Returns:
            Coefficient matrix for right qubit mapping constraints.
        """
        # Create sparse data container with default empty lists
        gate_mapping_right_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        # For each layer, process all operations that execute in parallel
        for layer_index in range(self.layer_count):
            # Calculate the timestep when operations in this layer execute
            spaced_timestep = layer_index * self.worst_spacing

            # For each operation in this layer
            for operation in self.gate_layers[layer_index]:
                # Get the operation index for variable indexing
                operation_index = self.operations.index(operation)

                # Get the logical qubits participating in this operation
                left_logical_qubit, right_logical_qubit = operation.qubits_participating

                for physical_edge_index in range(self.coupling_map.number_of_edges()):
                    # Build multidimensional index for gate execution variable (2D)
                    gate_execution_multi_index = (
                        operation_index,
                        physical_edge_index,
                    )
                    # Convert to flat index within gate execution variable space
                    gate_execution_ravel_index = np.ravel_multi_index(
                        gate_execution_multi_index, self.gate_execution_variables_shape
                    )
                    # Apply offset to position correctly in full decision variable vector
                    flattened_gate_execution_index = (
                        self.flat_mapping_variables_shape + gate_execution_ravel_index
                    )
                    # Add coefficient of 1 for gate execution variable
                    gate_mapping_right_sparse_data.add_coefficient(
                        1.0, constraint_index, flattened_gate_execution_index
                    )

                    # Get the physical edge (left and right physical qubits)
                    physical_edge = self.coupling_map_edges[physical_edge_index]
                    left_physical_qubit, right_physical_qubit = physical_edge

                    # Build multidimensional index for right qubit mapping variable
                    right_qubit_mapping_multi_index = (
                        spaced_timestep,
                        right_physical_qubit.index,
                        right_logical_qubit.index,
                    )
                    # Convert to flat index for right qubit mapping variable
                    right_qubit_mapping_ravel_index = np.ravel_multi_index(
                        right_qubit_mapping_multi_index, self.mapping_variables_shape
                    )
                    # Add coefficient of -1 for right qubit mapping variable
                    gate_mapping_right_sparse_data.add_coefficient(
                        -1.0, constraint_index, right_qubit_mapping_ravel_index
                    )

                    # Move to next constraint row after processing this operation-edge combination
                    constraint_index += 1

        # Build sparse coefficient matrix
        # Total constraints = final constraint_index (number of constraints processed)
        gate_mapping_right_constraint = self._build_sparse_coefficient_matrix(
            gate_mapping_right_sparse_data, constraint_index
        )
        return gate_mapping_right_constraint

    def _generate_gate_mapping_full_qubit_constraint(self):
        """Generate full qubit mapping constraint coefficients using McCormick relaxation.

        Enforces the constraint between gate execution and both qubit mapping variables.

        Returns:
            Coefficient matrix for full qubit mapping constraints.
        """
        # Create sparse data container with default empty lists
        gate_mapping_full_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        # For each layer, process all operations that execute in parallel
        for layer_index in range(self.layer_count):
            # Calculate the timestep when operations in this layer execute
            spaced_timestep = layer_index * self.worst_spacing

            # For each operation in this layer
            for operation in self.gate_layers[layer_index]:
                # Get the operation index for variable indexing
                operation_index = self.operations.index(operation)

                # Get the logical qubits participating in this operation
                left_logical_qubit, right_logical_qubit = operation.qubits_participating

                for physical_edge_index in range(self.coupling_map.number_of_edges()):
                    # Build multidimensional index for gate execution variable (2D)
                    gate_execution_multi_index = (
                        operation_index,
                        physical_edge_index,
                    )
                    # Convert to flat index within gate execution variable space
                    gate_execution_ravel_index = np.ravel_multi_index(
                        gate_execution_multi_index, self.gate_execution_variables_shape
                    )
                    # Apply offset to position correctly in full decision variable vector
                    flattened_gate_execution_index = (
                        self.flat_mapping_variables_shape + gate_execution_ravel_index
                    )
                    # Add coefficient of 1 for gate execution variable
                    gate_mapping_full_sparse_data.add_coefficient(
                        1.0, constraint_index, flattened_gate_execution_index
                    )

                    # Get the physical edge (left and right physical qubits)
                    physical_edge = self.coupling_map_edges[physical_edge_index]
                    left_physical_qubit, right_physical_qubit = physical_edge

                    # Build multidimensional index for left qubit mapping variable
                    left_qubit_mapping_multi_index = (
                        spaced_timestep,
                        left_physical_qubit.index,
                        left_logical_qubit.index,
                    )
                    # Convert to flat index for left qubit mapping variable
                    left_qubit_mapping_ravel_index = np.ravel_multi_index(
                        left_qubit_mapping_multi_index, self.mapping_variables_shape
                    )
                    # Add coefficient of -1 for left qubit mapping variable
                    gate_mapping_full_sparse_data.add_coefficient(
                        -1.0, constraint_index, left_qubit_mapping_ravel_index
                    )

                    # Build multidimensional index for right qubit mapping variable
                    right_qubit_mapping_multi_index = (
                        spaced_timestep,
                        right_physical_qubit.index,
                        right_logical_qubit.index,
                    )
                    # Convert to flat index for right qubit mapping variable
                    right_qubit_mapping_ravel_index = np.ravel_multi_index(
                        right_qubit_mapping_multi_index, self.mapping_variables_shape
                    )
                    # Add coefficient of -1 for right qubit mapping variable
                    gate_mapping_full_sparse_data.add_coefficient(
                        -1.0, constraint_index, right_qubit_mapping_ravel_index
                    )

                    # Move to next constraint row after processing this operation-edge combination
                    constraint_index += 1

        # Build sparse coefficient matrix
        # Total constraints = final constraint_index (number of constraints processed)
        gate_mapping_full_constraint = self._build_sparse_coefficient_matrix(
            gate_mapping_full_sparse_data, constraint_index
        )
        return gate_mapping_full_constraint

    def _generate_flow_condition_in_constraint(self):
        """Generate flow condition in constraint coefficients.

        Ensures qubit flow conservation for timesteps after the initial timestep.
        This constraint relates current mapping to previous timestep's movement variables.

        Returns:
            Coefficient matrix for flow condition in constraints.
        """
        # Create sparse data container with default empty lists
        flow_condition_in_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        # Calculate variable offset for movement variables in flattened decision vector
        movement_variables_offset = (
            self.flat_mapping_variables_shape + self.flat_gate_execution_variables_shape
        )

        # Skip first timestep (t=0) as it has no previous timestep
        for timestep in range(1, self.spaced_timesteps_count):
            previous_timestep = timestep - 1

            # For each logical qubit and each physical qubit position
            for logical_qubit in self.qubits:
                for physical_qubit in self.coupling_map.nodes:
                    # Current timestep mapping variable gets coefficient +1
                    current_mapping_multi_index = (
                        timestep,
                        physical_qubit.index,
                        logical_qubit.index,
                    )
                    current_mapping_ravel_index = np.ravel_multi_index(
                        current_mapping_multi_index, self.mapping_variables_shape
                    )
                    flow_condition_in_sparse_data.add_coefficient(
                        1.0, constraint_index, current_mapping_ravel_index
                    )

                    # Previous timestep self-movement variable gets coefficient -1
                    prev_self_movement_multi_index = (
                        previous_timestep,
                        logical_qubit.index,
                        physical_qubit.index,
                        physical_qubit.index,
                    )
                    prev_self_movement_ravel_index = np.ravel_multi_index(
                        prev_self_movement_multi_index, self.qubit_movement_shape
                    )
                    prev_self_movement_flat_index = (
                        movement_variables_offset + prev_self_movement_ravel_index
                    )
                    flow_condition_in_sparse_data.add_coefficient(
                        -1.0, constraint_index, prev_self_movement_flat_index
                    )

                    # Previous timestep incoming movement variables get coefficient -1
                    for neighbor_physical_qubit in self.coupling_map.neighbors(
                        physical_qubit
                    ):
                        prev_incoming_movement_multi_index = (
                            previous_timestep,
                            logical_qubit.index,
                            neighbor_physical_qubit.index,
                            physical_qubit.index,
                        )
                        prev_incoming_movement_ravel_index = np.ravel_multi_index(
                            prev_incoming_movement_multi_index,
                            self.qubit_movement_shape,
                        )
                        prev_incoming_movement_flat_index = (
                            movement_variables_offset
                            + prev_incoming_movement_ravel_index
                        )
                        flow_condition_in_sparse_data.add_coefficient(
                            -1.0, constraint_index, prev_incoming_movement_flat_index
                        )

                    # Move to next constraint row after processing this logical-physical qubit combination
                    constraint_index += 1

        # Build sparse coefficient matrix
        # Total constraints = final constraint_index (number of constraints processed)
        flow_condition_in_constraint = self._build_sparse_coefficient_matrix(
            flow_condition_in_sparse_data, constraint_index
        )
        return flow_condition_in_constraint

    def _generate_flow_condition_out_constraint(self):
        """Generate flow condition out constraint coefficients.

        Ensures qubit flow conservation for timesteps before the final timestep.
        This constraint relates current mapping to current timestep's movement variables.

        Returns:
            Coefficient matrix for flow condition out constraints.
        """
        # Create sparse data container with default empty lists
        flow_condition_out_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        # Calculate variable offset for movement variables in flattened decision vector
        movement_variables_offset = (
            self.flat_mapping_variables_shape + self.flat_gate_execution_variables_shape
        )

        # Include all timesteps to match notebook implementation
        for timestep in range(self.spaced_timesteps_count):
            # For each logical qubit and each physical qubit position
            for logical_qubit in self.qubits:
                for physical_qubit in self.coupling_map.nodes:
                    # Current timestep mapping variable gets coefficient +1
                    current_mapping_multi_index = (
                        timestep,
                        physical_qubit.index,
                        logical_qubit.index,
                    )
                    current_mapping_ravel_index = np.ravel_multi_index(
                        current_mapping_multi_index, self.mapping_variables_shape
                    )
                    flow_condition_out_sparse_data.add_coefficient(
                        1.0, constraint_index, current_mapping_ravel_index
                    )

                    # Current timestep self-movement variable gets coefficient -1
                    current_self_movement_multi_index = (
                        timestep,
                        logical_qubit.index,
                        physical_qubit.index,
                        physical_qubit.index,
                    )
                    current_self_movement_ravel_index = np.ravel_multi_index(
                        current_self_movement_multi_index, self.qubit_movement_shape
                    )
                    current_self_movement_flat_index = (
                        movement_variables_offset + current_self_movement_ravel_index
                    )
                    flow_condition_out_sparse_data.add_coefficient(
                        -1.0, constraint_index, current_self_movement_flat_index
                    )

                    # Current timestep outgoing movement variables get coefficient -1
                    for neighbor_physical_qubit in self.coupling_map.neighbors(
                        physical_qubit
                    ):
                        current_outgoing_movement_multi_index = (
                            timestep,
                            logical_qubit.index,
                            physical_qubit.index,
                            neighbor_physical_qubit.index,
                        )
                        current_outgoing_movement_ravel_index = np.ravel_multi_index(
                            current_outgoing_movement_multi_index,
                            self.qubit_movement_shape,
                        )
                        current_outgoing_movement_flat_index = (
                            movement_variables_offset
                            + current_outgoing_movement_ravel_index
                        )
                        flow_condition_out_sparse_data.add_coefficient(
                            -1.0, constraint_index, current_outgoing_movement_flat_index
                        )

                    # Move to next constraint row after processing this logical-physical qubit combination
                    constraint_index += 1

        # Build sparse coefficient matrix
        # Total constraints = final constraint_index (number of constraints processed)
        flow_condition_out_constraint = self._build_sparse_coefficient_matrix(
            flow_condition_out_sparse_data, constraint_index
        )
        return flow_condition_out_constraint

    def _generate_gate_execution_swap_constraint(self):
        """Generate gate execution swap constraint coefficients.

        Ensures that swaps can only happen between qubits that are executing a gate.
        This implements the constraint x^t_{p,i,j} = x^t_{q,j,i} where swaps are restricted
        to logical qubits participating in gate operations and their corresponding physical edges.

        Returns:
            Coefficient matrix for gate execution swap constraints.
        """
        # Create sparse data container with default empty lists
        gate_execution_swap_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        # Calculate variable offset for movement variables in flattened decision vector
        movement_variables_offset = (
            self.flat_mapping_variables_shape + self.flat_gate_execution_variables_shape
        )

        # For each layer, process all operations that execute in parallel
        for layer_index in range(self.layer_count):
            # Calculate the timestep when operations in this layer execute
            spaced_timestep = layer_index * self.worst_spacing

            # For each operation in this layer
            for operation in self.gate_layers[layer_index]:
                # Get the logical qubits participating in this operation
                left_logical_qubit, right_logical_qubit = operation.qubits_participating

                # For each physical edge in the coupling map
                for physical_edge_index in range(self.coupling_map.number_of_edges()):
                    physical_edge = self.coupling_map_edges[physical_edge_index]
                    left_physical_qubit, right_physical_qubit = physical_edge

                    # Build constraint: x^t_{p,i,j} - x^t_{q,j,i} = 0
                    # where p = left_physical_qubit, q = right_physical_qubit
                    # i = left_logical_qubit, j = right_logical_qubit

                    # Add coefficient +1 for movement from left physical to right physical
                    # for left logical qubit
                    forward_movement_multi_index = (
                        spaced_timestep,
                        left_logical_qubit.index,
                        left_physical_qubit.index,
                        right_physical_qubit.index,
                    )
                    forward_movement_ravel_index = np.ravel_multi_index(
                        forward_movement_multi_index, self.qubit_movement_shape
                    )
                    forward_movement_flat_index = (
                        movement_variables_offset + forward_movement_ravel_index
                    )
                    gate_execution_swap_sparse_data.add_coefficient(
                        1.0, constraint_index, forward_movement_flat_index
                    )

                    # Add coefficient -1 for movement from right physical to left physical
                    # for right logical qubit
                    backward_movement_multi_index = (
                        spaced_timestep,
                        right_logical_qubit.index,
                        right_physical_qubit.index,
                        left_physical_qubit.index,
                    )
                    backward_movement_ravel_index = np.ravel_multi_index(
                        backward_movement_multi_index, self.qubit_movement_shape
                    )
                    backward_movement_flat_index = (
                        movement_variables_offset + backward_movement_ravel_index
                    )
                    gate_execution_swap_sparse_data.add_coefficient(
                        -1.0, constraint_index, backward_movement_flat_index
                    )

                    # Move to next constraint row
                    constraint_index += 1

        # Build sparse coefficient matrix
        # Total constraints = final constraint_index (number of constraints processed)
        gate_execution_swap_constraint = self._build_sparse_coefficient_matrix(
            gate_execution_swap_sparse_data, constraint_index
        )
        return gate_execution_swap_constraint

    def _generate_non_gate_swap_constraint(self):
        """Generate non-gate swap constraint coefficients.

        Ensures movement balance for logical qubits NOT participating in gate executions.
        For each non-participating logical qubit and each physical edge, the sum of movements
        from left to right equals the sum of movements from right to left across all other logical qubits.

        Returns:
            Coefficient matrix for non-gate swap constraints.
        """
        # Create sparse data container with default empty lists
        non_gate_swap_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        # Calculate variable offset for movement variables in flattened decision vector
        movement_variables_offset = (
            self.flat_mapping_variables_shape + self.flat_gate_execution_variables_shape
        )

        # For each layer, process all operations that execute in parallel
        for layer_index in range(self.layer_count):
            # Calculate the timestep when operations in this layer execute
            spaced_timestep = layer_index * self.worst_spacing

            # Collect all logical qubits participating in ANY operation in this layer
            participating_qubits: set[quariadne.circuit.LogicalQubit] = set()
            for operation in self.gate_layers[layer_index]:
                participating_qubits.update(operation.qubits_participating)

            # Get all logical qubits that are NOT participating in any operation in this layer
            all_logical_qubits = set(self.qubits)
            non_participating_qubits = all_logical_qubits - participating_qubits

            # For each physical edge in the coupling map
            for physical_edge_index in range(self.coupling_map.number_of_edges()):
                physical_edge = self.coupling_map_edges[physical_edge_index]
                left_physical_qubit, right_physical_qubit = physical_edge

                # Build constraint: sum of movements involving all logical qubits
                # from left->right = sum of movements from right->left with other qubits

                # For each non-participating logical qubit
                for logical_qubit in non_participating_qubits:
                    # Add coefficient +1 for movement of this logical qubit with other qubit
                    # from left physical to right physical
                    forward_movement_multi_index = (
                        spaced_timestep,
                        logical_qubit.index,
                        left_physical_qubit.index,
                        right_physical_qubit.index,
                    )
                    forward_movement_ravel_index = np.ravel_multi_index(
                        forward_movement_multi_index, self.qubit_movement_shape
                    )
                    forward_movement_flat_index = (
                        movement_variables_offset + forward_movement_ravel_index
                    )
                    non_gate_swap_sparse_data.add_coefficient(
                        1.0, constraint_index, forward_movement_flat_index
                    )

                    # Add coefficient -1 for movement of this logical qubit with other qubit
                    # from right physical to left physical
                    backward_movement_multi_index = (
                        spaced_timestep,
                        logical_qubit.index,
                        right_physical_qubit.index,
                        left_physical_qubit.index,
                    )
                    backward_movement_ravel_index = np.ravel_multi_index(
                        backward_movement_multi_index, self.qubit_movement_shape
                    )
                    backward_movement_flat_index = (
                        movement_variables_offset + backward_movement_ravel_index
                    )
                    non_gate_swap_sparse_data.add_coefficient(
                        -1.0, constraint_index, backward_movement_flat_index
                    )

                # Move to next constraint row (one constraint per non-participating logical qubit per physical edge per timestep)
                constraint_index += 1

        # Build sparse coefficient matrix
        # Total constraints = final constraint_index (number of constraints processed)
        non_gate_swap_constraint = self._build_sparse_coefficient_matrix(
            non_gate_swap_sparse_data, constraint_index
        )
        return non_gate_swap_constraint

    def _generate_fixed_mapping_constraint(self):
        """Generate fixed mapping constraint coefficients.

        Constrains mapping variables to match the provided fixed mapping at operation timesteps.
        This constraint ensures that specific logical-to-physical qubit mappings are enforced
        when a fixed_mapping array is provided during router initialization.

        Returns:
            Coefficient matrix for fixed mapping constraints.
        """
        # Only generate constraints if fixed_mapping is provided
        if self.fixed_mapping is None:
            # Return empty constraint matrix if no fixed mapping
            empty_sparse_data = ConstraintMatrixSparseData()
            return self._build_sparse_coefficient_matrix(empty_sparse_data, 0)

        # Create sparse data container with default empty lists
        fixed_mapping_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        fixed_steps_count = self.fixed_mapping.shape[0]

        # Iterate through fixed timesteps (indexed by fixed_mapping rows)
        for fixed_timestep_index in range(fixed_steps_count):
            # Calculate the spaced timestep for this fixed mapping
            spaced_timestep = fixed_timestep_index * self.worst_spacing

            # Get the fixed mapping for this timestep
            timestep_fixed_mapping = self.fixed_mapping[fixed_timestep_index]
            # Find all non-zero positions in the fixed mapping for this timestep
            fixed_mapping_indices = np.argwhere(timestep_fixed_mapping)
            for physical_idx, logical_idx in fixed_mapping_indices:
                # Build multidimensional index for mapping variable
                mapping_multi_index = (
                    spaced_timestep,
                    physical_idx,
                    logical_idx,
                )

                # Convert to flat index for coefficient matrix
                flattened_mapping_index = np.ravel_multi_index(
                    mapping_multi_index, self.mapping_variables_shape
                )
                # Add coefficient of 1 for this mapping variable
                fixed_mapping_sparse_data.add_coefficient(
                    1.0, constraint_index, flattened_mapping_index
                )

                # Move to next constraint row for each fixed mapping position
                constraint_index += 1

        # Build sparse coefficient matrix
        # Total constraints = final constraint_index (number of constraints processed)
        fixed_mapping_constraint = self._build_sparse_coefficient_matrix(
            fixed_mapping_sparse_data, constraint_index
        )

        return fixed_mapping_constraint

    def _generate_fixed_edges_constraint(self):
        """Generate fixed edges constraint coefficients.

        Constrains gate_execution variables to match the provided fixed edges.
        This constraint ensures that specific operation-to-edge assignments are enforced
        when a fixed_edges list is provided during router initialisation.

        Returns:
            Coefficient matrix for fixed edges constraints.
        """
        # Only generate constraints if fixed_edges is provided
        if self.fixed_edges is None:
            # Return empty constraint matrix if no fixed edges
            empty_sparse_data = ConstraintMatrixSparseData()
            return self._build_sparse_coefficient_matrix(empty_sparse_data, 0)

        # Create sparse data container with default empty lists
        fixed_edges_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        layer_offset = 0
        # Iterate through each layer's edge assignments
        for layer_idx, layer_edge_assignments in enumerate(self.fixed_edges):
            # For each (operation_idx, edge_idx) assignment in this layer
            for operation_idx, edge_idx in layer_edge_assignments:
                offseted_operation_index = layer_offset + operation_idx
                # Build multidimensional index for gate execution variable (2D)
                gate_execution_multi_index = (
                    offseted_operation_index,
                    edge_idx,
                )

                # Convert to flat index within gate execution variable space
                gate_execution_ravel_index = np.ravel_multi_index(
                    gate_execution_multi_index, self.gate_execution_variables_shape
                )
                # Apply offset to position correctly in full decision variable vector
                flattened_gate_execution_index = (
                    self.flat_mapping_variables_shape + gate_execution_ravel_index
                )
                # Add coefficient of 1 for this gate execution variable
                fixed_edges_sparse_data.add_coefficient(
                    1.0, constraint_index, flattened_gate_execution_index
                )

                # Move to next constraint row for each fixed edge position
                constraint_index += 1
            layer_offset += len(self.gate_layers[layer_idx])

        # Build sparse coefficient matrix
        # Total constraints = final constraint_index (number of constraints processed)
        fixed_edges_constraint = self._build_sparse_coefficient_matrix(
            fixed_edges_sparse_data, constraint_index
        )

        return fixed_edges_constraint

    def _generate_all_constraints(self):
        """Generate all MILP constraints as scipy LinearConstraint objects.

        Iterates through the constraint registry and creates LinearConstraint objects
        for each constraint type with appropriate bounds.

        Returns:
            List of scipy.optimize.LinearConstraint objects for MILP formulation.
        """
        constraints = []

        for constraint_name in self.constraint_generators.keys():
            # Get the constraint generator function
            generate_constraint = self.constraint_generators[constraint_name]
            # Get the constraint bounds
            lower_bound = self.constraint_lower_bounds[constraint_name]
            upper_bound = self.constraint_upper_bounds[constraint_name]

            # Generate the coefficient matrix
            coefficient_matrix = generate_constraint()

            # Create a scipy LinearConstraint object
            linear_constraint = scipy.optimize.LinearConstraint(
                A=coefficient_matrix, lb=lower_bound, ub=upper_bound
            )

            constraints.append(linear_constraint)

        return constraints

    def _generate_optimisation_coefficients(self) -> np.ndarray:
        """Generate optimisation coefficients for the MILP objective function.

        Creates a coefficient vector for scipy.optimize.milp objective function.
        Assigns penalty coefficients to qubit movement variables to minimise routing overhead.

        Returns:
            Optimisation coefficient vector with shape (full_decision_variables_shape, ).
        """
        # Initialize coefficient vector for all decision variables
        optimisation_coefficients = np.zeros(
            self.full_decision_variables_shape, dtype=DEFAULT_CONSTRAINT_TYPE
        )

        # Calculate variable offset for movement variables in flattened decision vector
        movement_variables_offset = (
            self.flat_mapping_variables_shape + self.flat_gate_execution_variables_shape
        )

        # Iterate through all timesteps in the routing schedule
        for spaced_timestep in range(self.spaced_timesteps_count):
            # For each logical qubit and each possible qubit movement
            for logical_qubit in self.qubits:
                for from_physical_qubit in self.coupling_map.nodes:
                    for to_physical_qubit in self.coupling_map.nodes:
                        # Only penalise actual movement (not self-mapping)
                        if to_physical_qubit != from_physical_qubit:
                            # Build multidimensional index for qubit movement variable
                            qubit_movement_multi_index = (
                                spaced_timestep,
                                logical_qubit.index,
                                from_physical_qubit.index,
                                to_physical_qubit.index,
                            )
                            # Convert to flat index within movement variable space
                            qubit_movement_ravel_index = np.ravel_multi_index(
                                qubit_movement_multi_index, self.qubit_movement_shape
                            )
                            # Apply offset to position correctly in full decision variable vector
                            flattened_qubit_movement_index = (
                                movement_variables_offset + qubit_movement_ravel_index
                            )
                            # Set penalty coefficient for this movement variable
                            optimisation_coefficients[
                                flattened_qubit_movement_index
                            ] = QUBIT_MOVEMENT_PENALTY_COEFFICIENT

        return optimisation_coefficients

    def _run_milp(self) -> scipy.optimize.OptimizeResult:
        """Run the Mixed-Integer Linear Programming optimisation for qubit routing.

        Combines all constraints and the objective function to solve the MILP problem
        using scipy.optimize.milp for optimal qubit routing. Solution rounding is applied
        only when integrality constraints are used (MILP mode).

        Returns:
            OptimizeResult object containing the optimal solution and metadata.
        """
        # Generate all constraints as LinearConstraint objects
        constraints = self._generate_all_constraints()

        # Generate optimisation coefficients for objective function
        optimisation_coefficients = self._generate_optimisation_coefficients()

        # Set up variable bounds: all variables are binary (0 or 1)
        variables_lower_bound = np.full(
            self.full_decision_variables_shape, BINARY_VARIABLE_LOWER_BOUND
        )
        variables_upper_bound = np.full(
            self.full_decision_variables_shape, BINARY_VARIABLE_UPPER_BOUND
        )
        variables_bounds = scipy.optimize.Bounds(
            variables_lower_bound, variables_upper_bound
        )

        # Set integrality constraints: all variables are integer (binary)
        integrality_constraints = np.full(
            self.full_decision_variables_shape, self.integrality
        )

        # Configure solver options
        solver_options = HiGHSSolverOptions(
            time_limit=MILP_SOLVER_TIMEOUT_SECONDS, write_model_to_file=True
        )
        solver_options_dict = asdict(solver_options)

        # Solve the MILP problem
        milp_result = scipy.optimize.milp(
            c=optimisation_coefficients,
            integrality=integrality_constraints,
            bounds=variables_bounds,
            constraints=constraints,
            options=solver_options_dict,
        )

        # Check for timeout (status == 1 indicates iteration or time limit reached)
        if milp_result.status == 1:
            raise TimeoutError(
                f"HiGHS solver exceeded time limit of {MILP_SOLVER_TIMEOUT_SECONDS} seconds. "
                f"Solver message: {milp_result.message}"
            )
        # TODO: Round solution to handle numerical precision issues
        # This may be removed when transitioning to scipy.optimize.linprog
        # TODO: fix typing issue
        if self.integrality == INTEGER_VARIABLE_INTEGRALITY:
            milp_result.x = np.rint(milp_result.x)  # type: ignore
        elif self.integrality == CONTINUOUS_VARIABLE_INTEGRALITY:
            close_to_zero_mask = np.isclose(milp_result.x, 0)  # type: ignore
            milp_result.x[close_to_zero_mask] = 0  # type: ignore

        return milp_result

    def _reconstruct_variables(
        self,
        milp_result: scipy.optimize.OptimizeResult,
        variable_type: RoutingVariableType,
    ) -> np.ndarray:
        """Generic function to reconstruct variables from MILP solution using registry.

        Args:
            milp_result: The scipy MILP optimization result containing the solution vector.
            variable_type: Type of variables to reconstruct using RoutingVariableType enum.

        Returns:
            Array with the specified shape containing reconstructed variables.
        """
        variable_info = self.variable_types[variable_type]
        variable_shape = variable_info[VARIABLE_SHAPE]
        flat_shape = variable_info[VARIABLE_FLAT_SHAPE]
        start_offset = variable_info[VARIABLE_OFFSET]

        # Initialize variables array
        variables = np.zeros(variable_shape)

        # Extract variables from solution vector
        for variable_index in range(flat_shape):
            # Convert flat index to multidimensional position
            variable_position = np.unravel_index(variable_index, variable_shape)
            # Get value from solution vector at correct offset
            variables[variable_position] = milp_result.x[start_offset + variable_index]

        return variables

    def run(self) -> MilpRouterResult:
        """Run the MILP optimization and return processed results.

        Executes the Mixed-Integer Linear Programming optimization for qubit routing
        and reconstructs all variable matrices from the solution.

        Returns:
            MilpRouterResult containing the optimization result and reconstructed variable matrices.
        """
        # Run the MILP optimization
        milp_result = self._run_milp()

        # Reconstruct all variable matrices using generic function with registry
        mapping_variables = self._reconstruct_variables(
            milp_result, RoutingVariableType.MAPPING
        )
        gate_execution_variables = self._reconstruct_variables(
            milp_result, RoutingVariableType.GATE_EXECUTION
        )
        qubit_movement_variables = self._reconstruct_variables(
            milp_result, RoutingVariableType.QUBIT_MOVEMENT
        )

        # Return the complete result
        return MilpRouterResult(
            milp_result=milp_result,
            mapping_variables=mapping_variables,
            gate_execution_variables=gate_execution_variables,
            qubit_movement_variables=qubit_movement_variables,
            worst_spacing=self.worst_spacing,
        )
