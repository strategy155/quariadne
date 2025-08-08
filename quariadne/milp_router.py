import networkx as nx
import numpy as np

import quariadne.circuit
import scipy.optimize

DEFAULT_SHAPE_TYPE = np.uint32

# Constraint name constants
LOGICAL_UNIQUENESS_CONSTRAINT = "logical_uniqueness_constraint"
PHYSICAL_UNIQUENESS_CONSTRAINT = "physical_uniqueness_constraint"
GATE_EXECUTION_CONSTRAINT = "gate_execution_constraint"
GATE_MAPPING_CONSTRAINT = "gate_mapping_constraint"
GATE_MAPPING_LEFT_QUBIT_CONSTRAINT = "gate_mapping_left_qubit_constraint"
GATE_MAPPING_RIGHT_QUBIT_CONSTRAINT = "gate_mapping_right_qubit_constraint"
FLOW_CONDITION_IN_CONSTRAINT = "flow_condition_in_constraint"
FLOW_CONDITION_OUT_CONSTRAINT = "flow_condition_out_constraint"

# Constraint bound value constants
EQUALITY_CONSTRAINT_BOUND = 1
ZERO_CONSTRAINT_BOUND = 0
INEQUALITY_LOWER_BOUND = -np.inf


class MilpRouter:
    def _add_dummy_qubits(self):
        # we check the compatibility with the given hardware, and deal with it accordingly
        routed_circuit_qubit_count = len(self.routed_circuit.qubits)
        if routed_circuit_qubit_count < self.coupling_map.number_of_nodes():
            offset_index_dummy = routed_circuit_qubit_count
            dummy_count = self.qubit_count - offset_index_dummy
            dummy_logical_qubits = [
                quariadne.circuit.LogicalQubit(dummy_index)
                for dummy_index in range(offset_index_dummy, dummy_count)
            ]
            self.routed_circuit.qubits += dummy_logical_qubits

        elif routed_circuit_qubit_count > self.coupling_map.number_of_nodes():
            raise TypeError("We got more qubits than we can route!")

    def _get_two_qubit_operations(self):
        two_qubit_gate_operations = []
        # first we get rid of single qubit gate operations
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

        self.coupling_map_edges = list(self.coupling_map.edges())

        # Constraint generator methods registry
        self.constraint_generators = {
            LOGICAL_UNIQUENESS_CONSTRAINT: self.generate_logical_uniqueness_constraint,
            PHYSICAL_UNIQUENESS_CONSTRAINT: self.generate_physical_uniqueness_constraint,
            GATE_EXECUTION_CONSTRAINT: self.generate_gate_execution_constraint,
            GATE_MAPPING_CONSTRAINT: self.generate_gate_mapping_constraint,
            GATE_MAPPING_LEFT_QUBIT_CONSTRAINT: self.generate_gate_mapping_left_qubit_constraint,
            GATE_MAPPING_RIGHT_QUBIT_CONSTRAINT: self.generate_gate_mapping_right_qubit_constraint,
            FLOW_CONDITION_IN_CONSTRAINT: self.generate_flow_condition_in_constraint,
            FLOW_CONDITION_OUT_CONSTRAINT: self.generate_flow_condition_out_constraint,
        }

        # Constraint lower bounds registry
        self.constraint_lower_bounds = {
            LOGICAL_UNIQUENESS_CONSTRAINT: EQUALITY_CONSTRAINT_BOUND,
            PHYSICAL_UNIQUENESS_CONSTRAINT: EQUALITY_CONSTRAINT_BOUND,
            GATE_EXECUTION_CONSTRAINT: EQUALITY_CONSTRAINT_BOUND,
            GATE_MAPPING_CONSTRAINT: INEQUALITY_LOWER_BOUND,
            GATE_MAPPING_LEFT_QUBIT_CONSTRAINT: INEQUALITY_LOWER_BOUND,
            GATE_MAPPING_RIGHT_QUBIT_CONSTRAINT: INEQUALITY_LOWER_BOUND,
            FLOW_CONDITION_IN_CONSTRAINT: ZERO_CONSTRAINT_BOUND,
            FLOW_CONDITION_OUT_CONSTRAINT: ZERO_CONSTRAINT_BOUND,
        }

        # Constraint upper bounds registry
        self.constraint_upper_bounds = {
            LOGICAL_UNIQUENESS_CONSTRAINT: EQUALITY_CONSTRAINT_BOUND,
            PHYSICAL_UNIQUENESS_CONSTRAINT: EQUALITY_CONSTRAINT_BOUND,
            GATE_EXECUTION_CONSTRAINT: EQUALITY_CONSTRAINT_BOUND,
            GATE_MAPPING_CONSTRAINT: ZERO_CONSTRAINT_BOUND,
            GATE_MAPPING_LEFT_QUBIT_CONSTRAINT: ZERO_CONSTRAINT_BOUND,
            GATE_MAPPING_RIGHT_QUBIT_CONSTRAINT: ZERO_CONSTRAINT_BOUND,
            FLOW_CONDITION_IN_CONSTRAINT: ZERO_CONSTRAINT_BOUND,
            FLOW_CONDITION_OUT_CONSTRAINT: ZERO_CONSTRAINT_BOUND,
        }

    def route(self) -> scipy.optimize.OptimizeResult:
        pass

    def _initialize_coefficient_matrix(self) -> np.ndarray:
        """Initialize a coefficient matrix stub for MILP constraint construction.

        In MILP formulation, this creates the zero matrix that will be populated with
        constraint coefficients. See scipy.optimize.milp documentation.

        Args:
            constraint_count: Number of constraints (rows) in the coefficient matrix.

        Returns:
            Zero coefficient matrix with shape (constraint_count, full_decision_variables_shape).
        """
        coefficient_matrix_shape = self.full_decision_variables_shape
        coefficient_matrix = np.zeros(coefficient_matrix_shape)
        return coefficient_matrix

    def generate_logical_uniqueness_constraint(self):
        """Generate logical qubit uniqueness constraint coefficients.

        Ensures each logical qubit maps to exactly one physical qubit at each timestep.

        Returns:
            Coefficient matrix for logical qubit uniqueness constraints.
        """
        logical_uniqueness_per_timestep_constraints = []
        # Iterate through all timesteps in the routing schedule
        for timestep in range(self.spaced_timesteps_count):
            # Create constraint row for this timestep's logical qubit uniqueness
            logical_uniqueness_coefficients = self._initialize_coefficient_matrix()
            # For each logical qubit, ensure it maps to exactly one physical qubit
            for logical_qubit in self.routed_circuit.qubits:
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
        logical_uniqueness_constraint = np.concatenate(
            logical_uniqueness_per_timestep_constraints
        )
        return logical_uniqueness_constraint

    def generate_physical_uniqueness_constraint(self):
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
        physical_uniqueness_constraint = np.concatenate(
            physical_uniqueness_per_timestep_constraints
        )
        return physical_uniqueness_constraint

    def generate_gate_execution_constraint(self):
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
        gate_execution_constraint = np.concatenate(
            gate_execution_per_operation_constraints
        )
        return gate_execution_constraint

    def generate_gate_mapping_constraint(self):
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
        gate_mapping_constraint = np.concatenate(gate_mapping_per_operation_constraints)
        return gate_mapping_constraint

    def generate_gate_mapping_left_qubit_constraint(self):
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
                    left_physical_qubit,
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
        gate_mapping_left_constraint = np.concatenate(
            gate_mapping_left_per_operation_constraints
        )
        return gate_mapping_left_constraint

    def generate_gate_mapping_right_qubit_constraint(self):
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
                    right_physical_qubit,
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
        gate_mapping_right_constraint = np.concatenate(
            gate_mapping_right_per_operation_constraints
        )
        return gate_mapping_right_constraint

    def generate_flow_condition_in_constraint(self):
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
                            neighbor_physical_qubit,
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
        flow_condition_in_constraint = np.concatenate(
            flow_condition_in_per_timestep_constraints
        )
        return flow_condition_in_constraint

    def generate_flow_condition_out_constraint(self):
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

        # Skip last timestep as it has no next timestep to flow into
        for timestep in range(self.spaced_timesteps_count - 1):
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
                            neighbor_physical_qubit,
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
        flow_condition_out_constraint = np.concatenate(
            flow_condition_out_per_timestep_constraints
        )
        return flow_condition_out_constraint

    def generate_all_constraints(self):
        """Generate all MILP constraints as scipy LinearConstraint objects.

        Iterates through the constraint registry and creates LinearConstraint objects
        for each constraint type with appropriate bounds.

        Returns:
            List of scipy.optimize.LinearConstraint objects for MILP formulation.
        """
        constraints = []

        for constraint_name in self.constraint_generators.keys():
            # Get the constraint generator function
            generator = self.constraint_generators[constraint_name]
            # Get the constraint bounds
            lower_bound = self.constraint_lower_bounds[constraint_name]
            upper_bound = self.constraint_upper_bounds[constraint_name]

            # Generate the coefficient matrix
            coefficient_matrix = generator()

            # Create a scipy LinearConstraint object
            linear_constraint = scipy.optimize.LinearConstraint(
                A=coefficient_matrix, lb=lower_bound, ub=upper_bound
            )

            constraints.append(linear_constraint)

        return constraints

    def generate_objective_function(self):
        pass
