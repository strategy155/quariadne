import networkx as nx
import numpy as np
from dataclasses import dataclass
from enum import Enum

import quariadne.circuit
import scipy.optimize

DEFAULT_SHAPE_TYPE = np.uint32

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
class MilpRouterResult:
    """Result container for MILP router optimization with processed variable matrices.

    Contains the three groups of decision variables from the MILP solution in unflattened form:
    - mapping_variables: qubit-to-physical mapping at each timestep
    - gate_execution_variables: which gates execute on which physical edges
    - qubit_movement_variables: qubit movement between physical locations
    """

    milp_result: scipy.optimize.OptimizeResult
    mapping_variables: np.ndarray
    gate_execution_variables: np.ndarray
    qubit_movement_variables: np.ndarray


class MilpRouter:
    """Mixed-Integer Linear Programming router for quantum circuit qubit routing.

    Implements a MILP-based approach to solve the quantum circuit routing problem,
    where logical qubits must be mapped to physical qubits while respecting hardware
    coupling constraints. The router uses scipy.optimize.milp to find optimal
    qubit mappings and SWAP operations.

    The MILP formulation includes three types of decision variables:
    - Mapping variables: qubit-to-physical mapping at each timestep
    - Gate execution variables: which gates execute on which physical edges
    - Qubit movement variables: qubit movement between physical locations

    Attributes:
        coupling_map: NetworkX DiGraph representing hardware connectivity
        routed_circuit: Abstract quantum circuit to be routed
        qubit_count: Number of physical qubits available
        operations: List of two-qubit operations to be routed
        operation_count: Number of operations to route
        worst_spacing: Worst-case timesteps for token swapping
        spaced_timesteps_count: Total timesteps in routing schedule
    """

    def _add_dummy_qubits(self):
        """Add dummy logical qubits to match hardware qubit count.

        Ensures the circuit has the same number of logical qubits as physical qubits
        available on the hardware. This is required for the MILP formulation to work
        correctly as it assumes a bijective mapping between logical and physical qubits.

        Raises:
            TypeError: If the circuit has more logical qubits than available physical qubits.
        """
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
            raise TypeError("We got more qubits than we can route!")

    def _get_two_qubit_operations(self):
        """Extract two-qubit operations from the quantum circuit.

        Filters the circuit operations to include only two-qubit gates, which are the
        ones that require routing due to coupling map constraints. Single-qubit operations
        can be executed on any physical qubit without routing considerations.

        Returns:
            List of QuantumOperation objects that involve exactly two qubits.

        Raises:
            TypeError: If any operation involves more than two qubits.
        """
        two_qubit_gate_operations = []
        for operation in self.routed_circuit.operations:
            if len(operation.qubits_participating) == 2:
                two_qubit_gate_operations.append(operation)
            elif len(operation.qubits_participating) > 2:
                raise TypeError("We got much more qubits that we want!")

        return two_qubit_gate_operations

    def __init__(
        self,
        coupling_map: nx.DiGraph,
        quantum_circuit: quariadne.circuit.AbstractQuantumCircuit,
    ):
        """Initialize the MILP router with hardware topology and quantum circuit.

        Sets up all necessary data structures for the MILP formulation including:
        - Variable shapes and offsets for mapping, gate execution, and qubit movement
        - Constraint generation registry with bounds
        - Timestep calculations based on worst-case token swapping

        Args:
            coupling_map: NetworkX DiGraph representing the physical qubit connectivity
                         and allowed two-qubit gate operations on the hardware.
            quantum_circuit: Abstract quantum circuit representation containing logical
                           qubits and operations to be routed onto the hardware.
        """
        self.coupling_map = coupling_map
        self.qubit_count = coupling_map.number_of_nodes()
        self.routed_circuit = quantum_circuit

        self._add_dummy_qubits()

        self.operations = self._get_two_qubit_operations()
        self.operation_count = len(self.operations)

        # calculating the parameters of the circuit
        # worst spacing is the token swapping worst case $n^2$
        self.worst_spacing = self.qubit_count**2
        self.spaced_timesteps_count = self.operation_count * self.worst_spacing

        print(self.routed_circuit.qubits)
        # calculating the shapes of the decision variables
        self.qubit_movement_shape = (
            self.spaced_timesteps_count,
            self.qubit_count,
            self.qubit_count,
            self.qubit_count,
        )
        self.mapping_variables_shape = (
            self.spaced_timesteps_count,
            self.qubit_count,
            self.qubit_count,
        )
        self.gate_execution_variables_shape = (
            self.spaced_timesteps_count,
            self.operation_count,
            self.coupling_map.number_of_edges(),
        )

        self.flat_qubit_movement_shape = np.prod(
            self.qubit_movement_shape, dtype=DEFAULT_SHAPE_TYPE
        )
        self.flat_mapping_variables_shape = np.prod(
            self.mapping_variables_shape, dtype=DEFAULT_SHAPE_TYPE
        )
        self.flat_gate_execution_variables_shape = np.prod(
            self.gate_execution_variables_shape, dtype=DEFAULT_SHAPE_TYPE
        )
        self.full_decision_variables_shape = (
            self.flat_mapping_variables_shape
            + self.flat_gate_execution_variables_shape
            + self.flat_qubit_movement_shape
        )

        # Precalculated variable offsets
        self.mapping_variables_offset = 0
        self.gate_execution_variables_offset = self.flat_mapping_variables_shape
        self.qubit_movement_variables_offset = (
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
        }

    def _initialize_coefficient_matrix(self) -> np.ndarray:
        """Initialize a coefficient matrix stub for MILP constraint construction.

        In MILP formulation, this creates the zero vector that will be populated with
        constraint coefficients for a single constraint row.

        Returns:
            Zero coefficient vector with shape (full_decision_variables_shape,).
        """
        coefficient_matrix_shape = self.full_decision_variables_shape
        coefficient_matrix = np.zeros(coefficient_matrix_shape)
        return coefficient_matrix

    def _generate_logical_uniqueness_constraint(self):
        """Generate logical qubit uniqueness constraint coefficients.

        Ensures each logical qubit maps to exactly one physical qubit at each timestep.

        Returns:
            Coefficient matrix for logical qubit uniqueness constraints.
        """
        logical_uniqueness_per_timestep_constraints = []
        # Iterate through all timesteps in the routing schedule
        for timestep in range(self.spaced_timesteps_count):
            # For each logical qubit, ensure it maps to exactly one physical qubit
            for logical_qubit in self.routed_circuit.qubits:
                # Create constraint row for this timestep's logical qubit uniqueness
                logical_uniqueness_coefficients = self._initialize_coefficient_matrix()

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
                    # Set coefficient to 1 for this mapping variable
                    logical_uniqueness_coefficients[
                        flattened_logical_by_physical_index
                    ] = 1
                logical_uniqueness_per_timestep_constraints.append(
                    logical_uniqueness_coefficients
                )

        # Combine all timestep constraints into single matrix
        logical_uniqueness_constraint = np.stack(
            logical_uniqueness_per_timestep_constraints
        )
        return logical_uniqueness_constraint

    def _generate_physical_uniqueness_constraint(self):
        """Generate physical qubit uniqueness constraint coefficients.

        Ensures each physical qubit hosts exactly one logical qubit at each timestep.

        Returns:
            Coefficient matrix for physical qubit uniqueness constraints.
        """
        physical_uniqueness_per_timestep_constraints = []
        # Iterate through all timesteps in the routing schedule
        for timestep in range(self.spaced_timesteps_count):
            # For each physical qubit, ensure it hosts exactly one logical qubit
            for physical_qubit in self.coupling_map.nodes:
                # Create constraint row for this physical qubit's uniqueness
                physical_uniqueness_coefficients = self._initialize_coefficient_matrix()
                for logical_qubit in self.routed_circuit.qubits:
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
                    # Set coefficient to 1 for this mapping variable
                    physical_uniqueness_coefficients[
                        flattened_physical_by_logical_index
                    ] = 1
                physical_uniqueness_per_timestep_constraints.append(
                    physical_uniqueness_coefficients
                )

        # Combine all physical qubit constraints into single matrix
        physical_uniqueness_constraint = np.stack(
            physical_uniqueness_per_timestep_constraints
        )
        return physical_uniqueness_constraint

    def _generate_gate_execution_constraint(self):
        """Generate gate execution uniqueness constraint coefficients.

        Ensures each gate executes on exactly one physical edge at its assigned timestep.

        Returns:
            Coefficient matrix for gate execution uniqueness constraints.
        """
        gate_execution_per_operation_constraints = []

        # For each operation, ensure it executes on exactly one physical edge
        for operation_index in range(self.operation_count):
            # Create constraint row for this operation's gate execution uniqueness
            gate_execution_coefficients = self._initialize_coefficient_matrix()
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
                # Set coefficient to 1 for this gate execution variable
                gate_execution_coefficients[flattened_gate_execution_index] = 1

            gate_execution_per_operation_constraints.append(gate_execution_coefficients)

        # Combine all operation constraints into single matrix
        gate_execution_constraint = np.stack(gate_execution_per_operation_constraints)
        return gate_execution_constraint

    def _generate_gate_mapping_constraint(self):
        """Generate gate mapping constraint coefficients using McCormick relaxation.

        Enforces the basic constraint for gate execution variables.
        This is part of the McCormick relaxation for gate mapping constraints.

        Returns:
            Coefficient matrix for gate mapping constraints.
        """
        gate_mapping_per_operation_constraints = []

        # For each operation and each possible physical edge
        for operation_index in range(self.operation_count):
            # Calculate the timestep when this operation is scheduled to execute
            spaced_timestep = operation_index * self.worst_spacing

            for physical_edge_index in range(self.coupling_map.number_of_edges()):
                # Create constraint row for this operation-edge combination
                gate_mapping_coefficients = self._initialize_coefficient_matrix()

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
                # Set coefficient to 1 for gate execution variable
                gate_mapping_coefficients[flattened_gate_execution_index] = 1

                gate_mapping_per_operation_constraints.append(gate_mapping_coefficients)

        # Combine all operation-edge constraints into single matrix
        gate_mapping_constraint = np.stack(gate_mapping_per_operation_constraints)
        return gate_mapping_constraint

    def _generate_gate_mapping_left_qubit_constraint(self):
        """Generate left qubit mapping constraint coefficients using McCormick relaxation.

        Enforces the constraint between gate execution and left qubit mapping variables.
        This is part of the McCormick relaxation for gate mapping constraints.

        Returns:
            Coefficient matrix for left qubit mapping constraints.
        """
        gate_mapping_left_per_operation_constraints = []

        # For each operation and each possible physical edge
        for operation_index in range(self.operation_count):
            # Calculate the timestep when this operation is scheduled to execute
            spaced_timestep = operation_index * self.worst_spacing

            for physical_edge_index in range(self.coupling_map.number_of_edges()):
                # Create constraint row for this operation-edge combination
                gate_mapping_left_coefficients = self._initialize_coefficient_matrix()

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
                # Set coefficient to 1 for gate execution variable
                gate_mapping_left_coefficients[flattened_gate_execution_index] = 1

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
                # Set coefficient to -1 for left qubit mapping variable
                gate_mapping_left_coefficients[left_qubit_mapping_ravel_index] = -1

                gate_mapping_left_per_operation_constraints.append(
                    gate_mapping_left_coefficients
                )

        # Combine all operation-edge constraints into single matrix
        gate_mapping_left_constraint = np.stack(
            gate_mapping_left_per_operation_constraints
        )
        return gate_mapping_left_constraint

    def _generate_gate_mapping_right_qubit_constraint(self):
        """Generate right qubit mapping constraint coefficients using McCormick relaxation.

        Enforces the constraint between gate execution and right qubit mapping variables.
        This is part of the McCormick relaxation for gate mapping constraints.

        Returns:
            Coefficient matrix for right qubit mapping constraints.
        """
        gate_mapping_right_per_operation_constraints = []

        # For each operation and each possible physical edge
        for operation_index in range(self.operation_count):
            # Calculate the timestep when this operation is scheduled to execute
            spaced_timestep = operation_index * self.worst_spacing

            for physical_edge_index in range(self.coupling_map.number_of_edges()):
                # Create constraint row for this operation-edge combination
                gate_mapping_right_coefficients = self._initialize_coefficient_matrix()

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
                # Set coefficient to 1 for gate execution variable
                gate_mapping_right_coefficients[flattened_gate_execution_index] = 1

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
                # Set coefficient to -1 for right qubit mapping variable
                gate_mapping_right_coefficients[right_qubit_mapping_ravel_index] = -1

                gate_mapping_right_per_operation_constraints.append(
                    gate_mapping_right_coefficients
                )

        # Combine all operation-edge constraints into single matrix
        gate_mapping_right_constraint = np.stack(
            gate_mapping_right_per_operation_constraints
        )
        return gate_mapping_right_constraint

    def _generate_gate_mapping_full_qubit_constraint(self):
        """Generate full qubit mapping constraint coefficients using McCormick relaxation.

        Enforces the constraint between gate execution and both qubit mapping variables.

        Returns:
            Coefficient matrix for full qubit mapping constraints.
        """
        gate_mapping_full_per_operation_constraints = []

        # For each operation and each possible physical edge
        for operation_index in range(self.operation_count):
            # Calculate the timestep when this operation is scheduled to execute
            spaced_timestep = operation_index * self.worst_spacing

            for physical_edge_index in range(self.coupling_map.number_of_edges()):
                # Create constraint row for this operation-edge combination
                gate_mapping_full_coefficients = self._initialize_coefficient_matrix()

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
                # Set coefficient to 1 for gate execution variable
                gate_mapping_full_coefficients[flattened_gate_execution_index] = 1

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
                # Set coefficient to -1 for left qubit mapping variable
                gate_mapping_full_coefficients[left_qubit_mapping_ravel_index] = -1

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
                # Set coefficient to -1 for right qubit mapping variable
                gate_mapping_full_coefficients[right_qubit_mapping_ravel_index] = -1

                gate_mapping_full_per_operation_constraints.append(
                    gate_mapping_full_coefficients
                )

        # Combine all operation-edge constraints into single matrix
        gate_mapping_full_constraint = np.stack(
            gate_mapping_full_per_operation_constraints
        )
        return gate_mapping_full_constraint

    def _generate_flow_condition_in_constraint(self):
        """Generate flow condition in constraint coefficients.

        Ensures qubit flow conservation for timesteps after the initial timestep.
        This constraint relates current mapping to previous timestep's movement variables.

        Returns:
            Coefficient matrix for flow condition in constraints.
        """
        flow_condition_in_per_timestep_constraints = []

        # Calculate variable offset for movement variables in flattened decision vector
        movement_variables_offset = (
            self.flat_mapping_variables_shape + self.flat_gate_execution_variables_shape
        )

        # Skip first timestep (t=0) as it has no previous timestep
        for timestep in range(1, self.spaced_timesteps_count):
            previous_timestep = timestep - 1

            # For each logical qubit and each physical qubit position
            for logical_qubit in self.routed_circuit.qubits:
                for physical_qubit in self.coupling_map.nodes:
                    # Create constraint row for this logical-physical qubit combination
                    flow_condition_in_coefficients = (
                        self._initialize_coefficient_matrix()
                    )

                    # Current timestep mapping variable gets coefficient +1
                    current_mapping_multi_index = (
                        timestep,
                        physical_qubit.index,
                        logical_qubit.index,
                    )
                    current_mapping_ravel_index = np.ravel_multi_index(
                        current_mapping_multi_index, self.mapping_variables_shape
                    )
                    flow_condition_in_coefficients[current_mapping_ravel_index] = 1

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
                    flow_condition_in_coefficients[prev_self_movement_flat_index] = -1

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
                        flow_condition_in_coefficients[
                            prev_incoming_movement_flat_index
                        ] = -1

                    flow_condition_in_per_timestep_constraints.append(
                        flow_condition_in_coefficients
                    )

        # Combine all flow condition in constraints into single matrix
        flow_condition_in_constraint = np.stack(
            flow_condition_in_per_timestep_constraints
        )
        return flow_condition_in_constraint

    def _generate_flow_condition_out_constraint(self):
        """Generate flow condition out constraint coefficients.

        Ensures qubit flow conservation for timesteps before the final timestep.
        This constraint relates current mapping to current timestep's movement variables.

        Returns:
            Coefficient matrix for flow condition out constraints.
        """
        flow_condition_out_per_timestep_constraints = []

        # Calculate variable offset for movement variables in flattened decision vector
        movement_variables_offset = (
            self.flat_mapping_variables_shape + self.flat_gate_execution_variables_shape
        )

        # Include all timesteps to match notebook implementation
        for timestep in range(self.spaced_timesteps_count):
            # For each logical qubit and each physical qubit position
            for logical_qubit in self.routed_circuit.qubits:
                for physical_qubit in self.coupling_map.nodes:
                    # Create constraint row for this logical-physical qubit combination
                    flow_condition_out_coefficients = (
                        self._initialize_coefficient_matrix()
                    )

                    # Current timestep mapping variable gets coefficient +1
                    current_mapping_multi_index = (
                        timestep,
                        physical_qubit.index,
                        logical_qubit.index,
                    )
                    current_mapping_ravel_index = np.ravel_multi_index(
                        current_mapping_multi_index, self.mapping_variables_shape
                    )
                    flow_condition_out_coefficients[current_mapping_ravel_index] = 1

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
                    flow_condition_out_coefficients[
                        current_self_movement_flat_index
                    ] = -1

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
                        flow_condition_out_coefficients[
                            current_outgoing_movement_flat_index
                        ] = -1

                    flow_condition_out_per_timestep_constraints.append(
                        flow_condition_out_coefficients
                    )

        # Combine all flow condition out constraints into single matrix
        flow_condition_out_constraint = np.stack(
            flow_condition_out_per_timestep_constraints
        )
        return flow_condition_out_constraint

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

            print(constraint_name, coefficient_matrix.shape, lower_bound, upper_bound)

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
        optimisation_coefficients = self._initialize_coefficient_matrix()

        # Calculate variable offset for movement variables in flattened decision vector
        movement_variables_offset = (
            self.flat_mapping_variables_shape + self.flat_gate_execution_variables_shape
        )

        # Iterate through all timesteps in the routing schedule
        for spaced_timestep in range(self.spaced_timesteps_count):
            # For each logical qubit and each possible qubit movement
            for logical_qubit in self.routed_circuit.qubits:
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
        using scipy.optimize.milp for optimal qubit routing.

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
            self.full_decision_variables_shape, INTEGER_VARIABLE_INTEGRALITY
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
        milp_result.x = np.rint(milp_result.x)

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
        )
