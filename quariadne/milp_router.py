import typing
import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import quariadne.circuit
import scipy.optimize
import scipy.sparse
import qiskit.transpiler
from birkhoff import birkhoff_von_neumann_decomposition

DEFAULT_SHAPE_TYPE = np.int64
DEFAULT_CONSTRAINT_TYPE = np.float64

# Optimisation coefficient constants
QUBIT_MOVEMENT_PENALTY_COEFFICIENT = 0.5

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
        two_qubit_operations: list[quariadne.circuit.QuantumOperation],
        qubits: tuple[quariadne.circuit.LogicalQubit, ...],
        integrality=INTEGER_VARIABLE_INTEGRALITY,
        fixed_mapping: np.ndarray | None = None,
    ):
        """Initialize the MILP router with hardware topology, operations, and qubits.

        Sets up all necessary data structures for the MILP formulation including:
        - Variable shapes and offsets for mapping, gate execution, and qubit movement
        - Constraint generation registry with bounds
        - Timestep calculations based on worst-case token swapping

        Args:
            coupling_map: NetworkX DiGraph representing the physical qubit connectivity
                         and allowed two-qubit gate operations on the hardware.
            two_qubit_operations: List of two-qubit operations to be routed.
            qubits: Tuple of logical qubits to be mapped to physical qubits.
            integrality: Integrality constraint value for decision variables. Use
                        INTEGER_VARIABLE_INTEGRALITY (1) for MILP or 0 for LP relaxation.
            fixed_mapping: Optional numpy array specifying fixed mapping constraints.
                          Should have shape matching the mapping variables format.
        """

        self.integrality = integrality
        self.fixed_mapping = fixed_mapping
        self.coupling_map = coupling_map
        self.qubit_count = coupling_map.number_of_nodes()

        # Store operations directly
        self.operations = two_qubit_operations
        self.operation_count = len(self.operations)

        # Calculate worst spacing before adding dummy qubits
        # worst spacing is the token swapping worst case n^2
        self.worst_spacing = len(qubits) - 1

        # Add dummy qubits to match hardware qubit count
        self.qubits = self._add_dummy_qubits(qubits)

        self.spaced_timesteps_count = self.operation_count * self.worst_spacing

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
            self.spaced_timesteps_count,
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
            GATE_MAPPING_CONSTRAINT: self._generate_gate_mapping_constraint,
            GATE_MAPPING_LEFT_QUBIT_CONSTRAINT: self._generate_gate_mapping_left_qubit_constraint,
            GATE_MAPPING_RIGHT_QUBIT_CONSTRAINT: self._generate_gate_mapping_right_qubit_constraint,
            GATE_MAPPING_FULL_QUBIT_CONSTRAINT: self._generate_gate_mapping_full_qubit_constraint,
            FLOW_CONDITION_IN_CONSTRAINT: self._generate_flow_condition_in_constraint,
            FLOW_CONDITION_OUT_CONSTRAINT: self._generate_flow_condition_out_constraint,
            GATE_EXECUTION_SWAP_CONSTRAINT: self._generate_gate_execution_swap_constraint,
            NON_GATE_SWAP_CONSTRAINT: self._generate_non_gate_swap_constraint,
            FIXED_MAPPING_CONSTRAINT: self._generate_fixed_mapping_constraint,
        }

        # Constraint lower bounds registry
        self.constraint_lower_bounds = {
            LOGICAL_UNIQUENESS_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            PHYSICAL_UNIQUENESS_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            GATE_EXECUTION_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            GATE_MAPPING_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            GATE_MAPPING_LEFT_QUBIT_CONSTRAINT: OPEN_LOWER_CONSTRAINT_BOUND,
            GATE_MAPPING_RIGHT_QUBIT_CONSTRAINT: OPEN_LOWER_CONSTRAINT_BOUND,
            GATE_MAPPING_FULL_QUBIT_CONSTRAINT: MINUS_ONE_LOWER_BOUND,
            FLOW_CONDITION_IN_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            FLOW_CONDITION_OUT_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            GATE_EXECUTION_SWAP_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            NON_GATE_SWAP_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            FIXED_MAPPING_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
        }

        # Constraint upper bounds registry
        self.constraint_upper_bounds = {
            LOGICAL_UNIQUENESS_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            PHYSICAL_UNIQUENESS_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            GATE_EXECUTION_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
            GATE_MAPPING_CONSTRAINT: OPEN_UPPER_CONSTRAINT_BOUND,
            GATE_MAPPING_LEFT_QUBIT_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            GATE_MAPPING_RIGHT_QUBIT_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            GATE_MAPPING_FULL_QUBIT_CONSTRAINT: OPEN_UPPER_CONSTRAINT_BOUND,
            FLOW_CONDITION_IN_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            FLOW_CONDITION_OUT_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            GATE_EXECUTION_SWAP_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            NON_GATE_SWAP_CONSTRAINT: ZERO_EQUALITY_CONSTRAINT_BOUND,
            FIXED_MAPPING_CONSTRAINT: ONE_EQUALITY_CONSTRAINT_BOUND,
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

        Returns:
            Coefficient matrix for gate execution uniqueness constraints.
        """
        # Create sparse data container with default empty lists
        gate_execution_sparse_data = ConstraintMatrixSparseData()

        constraint_index = 0

        # For each operation, ensure it executes on exactly one physical edge
        for operation_index in range(self.operation_count):
            # Calculate the timestep when this operation is scheduled to execute
            spaced_timestep = operation_index * self.worst_spacing

            # Consider all possible physical edges where the gate could execute
            for physical_edge_index in range(self.coupling_map.number_of_edges()):
                # Build multidimensional index for gate execution variable
                gate_execution_multi_index = (
                    spaced_timestep,
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

        # For each operation and each possible physical edge
        for operation_index in range(self.operation_count):
            # Calculate the timestep when this operation is scheduled to execute
            spaced_timestep = operation_index * self.worst_spacing

            for physical_edge_index in range(self.coupling_map.number_of_edges()):
                # Build multidimensional index for gate execution variable
                gate_execution_multi_index = (
                    spaced_timestep,
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

        # For each operation and each possible physical edge
        for operation_index in range(self.operation_count):
            # Calculate the timestep when this operation is scheduled to execute
            spaced_timestep = operation_index * self.worst_spacing

            for physical_edge_index in range(self.coupling_map.number_of_edges()):
                # Build multidimensional index for gate execution variable
                gate_execution_multi_index = (
                    spaced_timestep,
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

                # Get the logical qubits participating in this operation
                left_logical_qubit, right_logical_qubit = self.operations[
                    operation_index
                ].qubits_participating
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

        # For each operation and each possible physical edge
        for operation_index in range(self.operation_count):
            # Calculate the timestep when this operation is scheduled to execute
            spaced_timestep = operation_index * self.worst_spacing

            for physical_edge_index in range(self.coupling_map.number_of_edges()):
                # Build multidimensional index for gate execution variable
                gate_execution_multi_index = (
                    spaced_timestep,
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

                # Get the logical qubits participating in this operation
                left_logical_qubit, right_logical_qubit = self.operations[
                    operation_index
                ].qubits_participating

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

        # For each operation and each possible physical edge
        for operation_index in range(self.operation_count):
            # Calculate the timestep when this operation is scheduled to execute
            spaced_timestep = operation_index * self.worst_spacing

            for physical_edge_index in range(self.coupling_map.number_of_edges()):
                # Build multidimensional index for gate execution variable
                gate_execution_multi_index = (
                    spaced_timestep,
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

                # Get the logical qubits participating in this operation
                left_logical_qubit, right_logical_qubit = self.operations[
                    operation_index
                ].qubits_participating

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

        # For each operation timestep where gate execution occurs
        for operation_index in range(self.operation_count):
            # Calculate the timestep when this operation is scheduled to execute
            spaced_timestep = operation_index * self.worst_spacing

            # Get the logical qubits participating in this operation
            left_logical_qubit, right_logical_qubit = self.operations[
                operation_index
            ].qubits_participating

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

        # For each operation timestep where gate execution occurs
        for operation_index in range(self.operation_count):
            # Calculate the timestep when this operation is scheduled to execute
            spaced_timestep = operation_index * self.worst_spacing

            # Get the logical qubits participating in this operation
            participating_qubits = set(
                self.operations[operation_index].qubits_participating
            )

            # Get all logical qubits that are NOT participating in this operation
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

        # Iterate through operation timesteps (not all spaced timesteps)
        for operation_index in range(fixed_steps_count):
            # Calculate the timestep when this operation is scheduled to execute
            spaced_timestep = operation_index * self.worst_spacing

            # Get the fixed mapping for this timestep
            timestep_fixed_mapping = self.fixed_mapping[operation_index]
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

        # Solve the MILP problem
        milp_result = scipy.optimize.milp(
            c=optimisation_coefficients,
            integrality=integrality_constraints,
            bounds=variables_bounds,
            constraints=constraints,
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


class IlpRouter:
    """Integer Linear Programming router for quantum circuit qubit routing.

    High-level router that handles the complete routing process including running
    the optimisation, processing results, and extracting initial mappings and swap operations.
    Uses MilpScipyRouter internally for the mathematical optimisation.

    This class provides the main interface for quantum circuit routing and automatically
    processes the results into usable routing information.

    Attributes:
        result: The routing result containing all variable matrices
    """

    def __init__(
        self,
        coupling_map: nx.DiGraph,
        quantum_circuit: quariadne.circuit.AbstractQuantumCircuit,
    ):
        """Initialize the ILP router and automatically run the routing.

        Args:
            coupling_map: NetworkX DiGraph representing the physical qubit connectivity
            quantum_circuit: Abstract quantum circuit representation to be routed
        """
        two_qubit_operations = quantum_circuit.get_two_qubit_operations()
        qubits = quantum_circuit.qubits

        milp_router = MilpScipyRouter(
            coupling_map=coupling_map,
            two_qubit_operations=two_qubit_operations,
            qubits=qubits,
            integrality=INTEGER_VARIABLE_INTEGRALITY,
            fixed_mapping=None,
        )
        self.result = milp_router.run()

    def get_initial_mapping(
        self,
    ) -> dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]:
        """Calculate initial mapping from mapping variables at timestep 0.

        Extracts the mapping at the first timestep (t=0) and returns a dictionary
        mapping logical qubits to physical qubits for the initial state.

        Returns:
            Dictionary mapping LogicalQubit objects to PhysicalQubit objects
            representing the initial qubit mapping.
        """
        # Extract the mapping matrix at timestep 0
        initial_timestep_mapping = self.result.mapping_variables[0]

        # Find all non-zero positions (physical_idx, logical_idx) where mapping exists
        initial_mapping_indices = np.argwhere(initial_timestep_mapping)

        # Convert numpy array to Python list for iteration
        # tolist() ensures we get Python integers rather than numpy.int64 objects
        # which are required for the PhysicalQubit and LogicalQubit constructors
        initial_mapping_indices_list = initial_mapping_indices.tolist()

        # Build the mapping dictionary (logical → physical, matching notebook implementation)
        mapping_dict = {}
        for physical_idx, logical_idx in initial_mapping_indices_list:
            physical_qubit = quariadne.circuit.PhysicalQubit(physical_idx)
            logical_qubit = quariadne.circuit.LogicalQubit(logical_idx)
            mapping_dict[logical_qubit] = physical_qubit

        return mapping_dict

    def get_inserted_swaps(self) -> dict[int, list[quariadne.circuit.PhysicalSwap]]:
        """Extract swap operations from qubit movement variables.

        TODO: Implement this operation e2e, now this is too raw.
        Analyses the qubit movement variables to identify SWAP operations that occur
        after each operation timestep. SWAP operations are detected by finding pairs of
        qubits that exchange positions between physical locations.

        Returns:
            Dictionary mapping operation indices (0-based) to lists of PhysicalSwap objects.
            Each swap pair is represented as a PhysicalSwap containing two PhysicalQubit objects.

        Example:
            {
                0: [],  # No swaps after operation 0
                1: [PhysicalSwap(PhysicalQubit(0), PhysicalQubit(1))],  # SWAP between physical qubits 0 and 1 after operation 1
                2: [PhysicalSwap(PhysicalQubit(2), PhysicalQubit(3)), PhysicalSwap(PhysicalQubit(0), PhysicalQubit(4))]  # Two SWAPs after operation 2
            }
        """
        swap_pairs_by_operation: defaultdict = defaultdict(list)

        # Find all non-zero movement variables where actual movement occurs
        defined_movements = np.argwhere(self.result.qubit_movement_variables)

        for movement in defined_movements:
            timestep, logical_qubit, from_qubit, to_qubit = movement.tolist()

            # Only consider actual movements (not self-assignments)
            if from_qubit != to_qubit:
                # Convert timestep to operation index using worst_spacing division
                # The +1 is an offset accounting for the fact that swaps happen AFTER respective operations
                operation_index = timestep // self.result.worst_spacing + 1

                # Create PhysicalSwap with proper qubit objects
                physical_qubit_from = quariadne.circuit.PhysicalQubit(from_qubit)
                physical_qubit_to = quariadne.circuit.PhysicalQubit(to_qubit)
                swap_pair = quariadne.circuit.PhysicalSwap(
                    physical_qubit_from, physical_qubit_to
                )

                # Get current operation's swap list
                current_operation_swaps = swap_pairs_by_operation[operation_index]

                # Add swap pair if not already present (avoids duplicates through PhysicalSwap equality)
                if swap_pair not in current_operation_swaps:
                    current_operation_swaps.append(swap_pair)

        return swap_pairs_by_operation


class LpRouter:
    """Linear Programming router for quantum circuit qubit routing using iterative mapping recovery.

    This router uses continuous variables (integrality=0) instead of integer variables and employs
    an iterative approach with Birkhoff-von Neumann decomposition to extract permutation mappings
    from doubly stochastic matrices. The process follows a bootstrap phase followed by iterative
    mapping recovery to generate a full scheme of mappings and sequence of swaps.

    Attributes:
        coupling_map: NetworkX DiGraph representing physical qubit connectivity
        qubits: Tuple of logical qubits to be routed
        two_qubit_operations: List of two-qubit operations to be routed
        mappings: List of extracted mappings from iterative process
        permutations: List of permutation matrices corresponding to mappings
        swaps_by_operation: Dictionary mapping operation indices to swap lists
    """

    def __init__(
        self,
        coupling_map: nx.DiGraph,
        quantum_circuit: quariadne.circuit.AbstractQuantumCircuit,
    ):
        """Initialize the LP router and automatically run the routing.

        Args:
            coupling_map: NetworkX DiGraph representing the physical qubit connectivity
            quantum_circuit: Abstract quantum circuit representation to be routed
        """
        self.coupling_map = coupling_map
        self.qubits = quantum_circuit.qubits
        self.two_qubit_operations = quantum_circuit.get_two_qubit_operations()
        self.mappings: list[
            dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]
        ] = []
        self.permutations: list[np.ndarray] = []
        self.swaps_by_operation: dict[int, list[quariadne.circuit.PhysicalSwap]] = {}

        # Run routing automatically
        self._run()

    def _get_initial_permutation(self) -> np.ndarray:
        """Get initial permutation using LP bootstrap phase.

        Uses Linear Programming (integrality=0) without constraints to get the initial
        mapping through optimisation. Extracts a compatible permutation using Birkhoff decomposition.

        Returns:
            Initial permutation matrix
        """
        # Get first two-qubit operation
        first_two_qubit_operation = self.two_qubit_operations[0]

        # Run LP optimisation with continuous variables and no fixed mapping
        lp_router = MilpScipyRouter(
            coupling_map=self.coupling_map,
            two_qubit_operations=self.two_qubit_operations,
            qubits=self.qubits,
            integrality=CONTINUOUS_VARIABLE_INTEGRALITY,
            fixed_mapping=None,
        )
        lp_result = lp_router.run()

        # Extract compatible permutation for first two-qubit operation
        bootstrap_mapping_variables = lp_result.mapping_variables[0]
        initial_permutation = self._extract_permutation_for_operation(
            bootstrap_mapping_variables, first_two_qubit_operation
        )

        return initial_permutation

    def _get_next_permutation(
        self,
        current_permutation: np.ndarray,
        current_operation: quariadne.circuit.QuantumOperation,
        remaining_operations: list[quariadne.circuit.QuantumOperation],
    ) -> np.ndarray:
        """Get next permutation by running LP on remaining operations with fixed current permutation.

        Args:
            current_permutation: Current permutation matrix to fix as constraint
            current_operation: Operation that needs to be made executable
            remaining_operations: List of remaining two-qubit operations to route

        Returns:
            Next permutation matrix compatible with the current operation
        """
        # Run LP on remaining operations with current permutation fixed
        fixed_permutation_array = current_permutation[np.newaxis, ...]

        lp_router = MilpScipyRouter(
            coupling_map=self.coupling_map,
            two_qubit_operations=remaining_operations,
            qubits=self.qubits,
            integrality=CONTINUOUS_VARIABLE_INTEGRALITY,
            fixed_mapping=fixed_permutation_array,
        )
        lp_result = lp_router.run()

        # Extract next permutation at first operation timestep
        next_mapping_timestep = lp_result.worst_spacing
        next_mapping_variables = lp_result.mapping_variables[next_mapping_timestep]

        next_permutation = self._extract_permutation_for_operation(
            next_mapping_variables, current_operation
        )
        return next_permutation

    def _create_physical_to_logical_mapping(
        self, permutation_matrix: np.ndarray
    ) -> dict[quariadne.circuit.PhysicalQubit, quariadne.circuit.LogicalQubit]:
        """Create a mapping from physical to logical qubits from a permutation matrix.

        Args:
            permutation_matrix: 2D numpy array representing a permutation matrix

        Returns:
            Dictionary mapping PhysicalQubit objects to LogicalQubit objects.
        """
        physical_to_logical_mapping = {}
        for physical_idx, logical_idx in np.argwhere(permutation_matrix):
            physical_qubit = quariadne.circuit.PhysicalQubit(physical_idx)
            logical_qubit = quariadne.circuit.LogicalQubit(logical_idx)
            physical_to_logical_mapping[physical_qubit] = logical_qubit
        return physical_to_logical_mapping

    def _is_compatible_with_operation(
        self,
        permutation_matrix: np.ndarray,
        operation: quariadne.circuit.QuantumOperation,
    ) -> bool:
        """Check if a permutation matrix is compatible with a two-qubit operation.

        Args:
            permutation_matrix: 2D numpy array representing a permutation matrix
            operation: QuantumOperation that requires connectivity between its participating qubits

        Returns:
            True if the permutation allows the operation's qubits to be connected, False otherwise.
        """
        # Create the mapping from physical to logical qubits for this permutation
        physical_to_logical_mapping = self._create_physical_to_logical_mapping(
            permutation_matrix
        )

        # Relabel the coupling map to get logical connectivity
        logical_connectivity = nx.relabel_nodes(
            self.coupling_map, physical_to_logical_mapping
        )

        # Check if the operation's participating qubits are connected
        left_logical, right_logical = operation.qubits_participating

        return logical_connectivity.has_edge(left_logical, right_logical)

    def _build_mapping_from_permutation(
        self, permutation_matrix: np.ndarray
    ) -> dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]:
        """Build a logical-to-physical mapping dictionary from a permutation matrix.

        Args:
            permutation_matrix: 2D numpy array representing a permutation matrix

        Returns:
            Dictionary mapping LogicalQubit objects to PhysicalQubit objects.
        """
        mapping_dict = {}
        for physical_idx, logical_idx in np.argwhere(permutation_matrix):
            physical_qubit = quariadne.circuit.PhysicalQubit(physical_idx)
            logical_qubit = quariadne.circuit.LogicalQubit(logical_idx)
            mapping_dict[logical_qubit] = physical_qubit
        return mapping_dict

    def _extract_permutation_for_operation(
        self,
        mapping_variables: np.ndarray,
        operation: quariadne.circuit.QuantumOperation,
    ) -> np.ndarray:
        """Extract permutation matrix for a given operation using Birkhoff decomposition.

        Uses Birkhoff-von Neumann decomposition to convert doubly stochastic matrix to
        permutations and selects the first permutation that satisfies connectivity
        requirements for the given operation.

        Args:
            mapping_variables: 2D numpy array representing doubly stochastic mapping matrix
            operation: QuantumOperation that requires connectivity checking

        Returns:
            2D numpy array representing the permutation matrix.

        Raises:
            ValueError: If no compatible permutation found for the operation.
        """
        # Perform Birkhoff-von Neumann decomposition
        coefficient_matrix_pairs = birkhoff_von_neumann_decomposition(mapping_variables)

        # Sort permutations by coefficient in descending order
        sorted_permutations = sorted(
            coefficient_matrix_pairs,
            key=lambda coefficient_matrix_pair: coefficient_matrix_pair[0],
            reverse=True,
        )

        # Find the first permutation that satisfies connectivity requirements
        for coefficient, permutation_matrix in sorted_permutations:
            if self._is_compatible_with_operation(permutation_matrix, operation):
                return permutation_matrix

        # If no compatible permutation found, raise an error
        raise ValueError(f"No compatible permutation found for operation {operation}")

    def _mapping_to_permutation_matrix(
        self,
        mapping: dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit],
    ) -> np.ndarray:
        """Convert logical-to-physical mapping dictionary to permutation matrix.

        Args:
            mapping: Dictionary mapping LogicalQubit objects to PhysicalQubit objects

        Returns:
            2D numpy array representing the permutation matrix
        """
        qubit_count = len(mapping)
        permutation_matrix = np.zeros((qubit_count, qubit_count))

        for logical_qubit, physical_qubit in mapping.items():
            permutation_matrix[physical_qubit.index, logical_qubit.index] = 1.0

        return permutation_matrix

    def _accumulate_edge_swaps(
        self, qubit_movement_variables: np.ndarray
    ) -> dict[SwapEdge, float]:
        """Accumulate movement values for each edge from qubit movement variables.

        Args:
            qubit_movement_variables: Array of qubit movement variables from router result

        Returns:
            Dictionary mapping edge tuples (from_qubit, to_qubit) to accumulated movement values
        """
        swap_edges: dict[SwapEdge, float] = defaultdict(float)
        defined_movements = np.argwhere(qubit_movement_variables)

        for movement in defined_movements:
            timestep, logical_qubit, from_qubit, to_qubit = movement.tolist()

            # Only consider actual movements (not self-assignments)
            if from_qubit != to_qubit:
                edge_key = (from_qubit, to_qubit)
                movement_value = qubit_movement_variables[
                    timestep, logical_qubit, from_qubit, to_qubit
                ]
                swap_edges[edge_key] += movement_value

        return swap_edges

    def _create_swaps_from_edges(
        self, swap_edges: dict[SwapEdge, float]
    ) -> list[quariadne.circuit.PhysicalSwap]:
        """Create list of PhysicalSwap objects from accumulated edge movements.

        Args:
            swap_edges: Dictionary mapping edge tuples to accumulated movement values

        Returns:
            List of PhysicalSwap objects for movements approximately equal to 1.0
        """
        swaps = []
        for (from_qubit, to_qubit), accumulated_movement_value in swap_edges.items():
            # Only include movements approximately equal to 1.0
            if np.isclose(accumulated_movement_value, 1.0):
                physical_qubit_from = quariadne.circuit.PhysicalQubit(from_qubit)
                physical_qubit_to = quariadne.circuit.PhysicalQubit(to_qubit)
                swap = quariadne.circuit.PhysicalSwap(
                    physical_qubit_from, physical_qubit_to
                )

                # Add swap if not already present (avoids duplicates through PhysicalSwap equality)
                if swap not in swaps:
                    swaps.append(swap)

        return swaps

    def _get_swaps_between_two_permutations(
        self,
        previous_permutation: np.ndarray,
        current_permutation: np.ndarray,
        previous_operation: quariadne.circuit.QuantumOperation,
        current_operation: quariadne.circuit.QuantumOperation,
    ) -> list[quariadne.circuit.PhysicalSwap]:
        """Calculate swap operations between two consecutive permutations.

        Runs a constrained routing problem with exactly two operations and two fixed
        permutations to recover the minimal sequence of swaps needed to transition
        between them.

        Args:
            previous_permutation: Permutation matrix for the previous operation
            current_permutation: Permutation matrix for the current operation
            previous_operation: The previous quantum operation
            current_operation: The current quantum operation

        Returns:
            List of PhysicalSwap operations needed to transition between permutations
        """
        two_step_permutations = np.array([previous_permutation, current_permutation])
        two_step_operations = [previous_operation, current_operation]
        # Run router with both operations and fixed mappings
        lp_router = MilpScipyRouter(
            coupling_map=self.coupling_map,
            two_qubit_operations=two_step_operations,
            qubits=self.qubits,
            integrality=CONTINUOUS_VARIABLE_INTEGRALITY,
            fixed_mapping=two_step_permutations,
        )
        lp_result = lp_router.run()

        # Phase 1: Accumulate edge swaps
        swap_edges = self._accumulate_edge_swaps(lp_result.qubit_movement_variables)

        # Phase 2: Create list of swaps
        swaps = self._create_swaps_from_edges(swap_edges)

        return swaps

    def _run(self) -> None:
        """Run the iterative LP routing process to generate sequence of mappings and swaps.

        Implements the iterative routing algorithm:
        1. Bootstrap: get initial permutation using LP and Birkhoff decomposition
        2. Iterate through two-qubit operations:
           - Check compatibility with current permutation
           - If incompatible, run LP on remaining operations to get next permutation
           - Calculate swaps between previous and current permutation
           - Pop the processed operation
        """
        # Make a working copy of operations list
        remaining_operations = self.two_qubit_operations.copy()

        # Bootstrap phase: get initial permutation
        current_permutation = self._get_initial_permutation()
        initial_mapping = self._build_mapping_from_permutation(current_permutation)
        self.mappings = [initial_mapping]
        self.permutations = [current_permutation]

        # Iterative phase: process operations one by one
        for operation_index in range(1, len(self.two_qubit_operations)):
            routed_operation = self.two_qubit_operations[operation_index]
            previous_operation = self.two_qubit_operations[operation_index - 1]
            previous_permutation = current_permutation

            # Check if operation is incompatible with current permutation
            if not self._is_compatible_with_operation(
                current_permutation, routed_operation
            ):
                # Two-qubit operation is not executable, get next permutation
                next_permutation = self._get_next_permutation(
                    current_permutation, routed_operation, remaining_operations
                )
                next_mapping = self._build_mapping_from_permutation(next_permutation)

                # Calculate swaps between previous and next permutation
                swaps = self._get_swaps_between_two_permutations(
                    previous_permutation,
                    next_permutation,
                    previous_operation,
                    routed_operation,
                )
                self.swaps_by_operation[operation_index] = swaps

                # Update current permutation and add mapping to list
                self.mappings.append(next_mapping)
                self.permutations.append(next_permutation)
                current_permutation = next_permutation
            else:
                # Operation is compatible, no swaps needed
                self.swaps_by_operation[operation_index] = []

            # Pop the first operation from remaining operations
            remaining_operations.pop(0)

    def get_initial_mapping(
        self,
    ) -> dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]:
        """Get initial mapping from the first permutation.

        Returns:
            Dictionary mapping LogicalQubit objects to PhysicalQubit objects
            representing the initial qubit mapping.
        """
        return self.mappings[0]

    def get_inserted_swaps(self) -> dict[int, list[quariadne.circuit.PhysicalSwap]]:
        """Get swap operations inserted after each operation.

        Returns:
            Dictionary mapping operation indices to lists of PhysicalSwap objects.
            Each swap list contains swaps needed after executing the operation at that index.
        """
        return self.swaps_by_operation
