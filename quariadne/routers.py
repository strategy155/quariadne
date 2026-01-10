from abc import ABC, abstractmethod
from collections import defaultdict

import networkx as nx
import numpy as np

from birkhoff import birkhoff_von_neumann_decomposition
import scipy.optimize

import quariadne.circuit
from quariadne.milp import (
    MilpScipyRouter,
    INTEGER_VARIABLE_INTEGRALITY,
    CONTINUOUS_VARIABLE_INTEGRALITY,
    SwapEdge,
    create_gate_layers,
    GateLayer,
)

# Small constant added to bipartite matching weights to ensure feasibility:
# - On forcing edges: ensures self-matching is always possible
# - On real edges: allows matching edges with 0.0 execution weight
BIPARTITE_FORCING_TERM = 1e-6

# Type alias for operation-to-edge assignment: (operation_index, edge_index)
OperationEdgeAssignment = tuple[int, int]


def accumulate_edge_swaps(
    qubit_movement_variables: np.ndarray,
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


def create_swaps_from_edges(
    swap_edges: dict[SwapEdge, float],
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


class Router(ABC):
    """Base class for quantum circuit routing algorithms.

    Provides common initialisation and setup for all router implementations.
    Subclasses implement specific routing algorithms (MILP, LP with Birkhoff, LP with edges).

    Attributes:
        circuit: The quantum circuit to route
        coupling_map: NetworkX DiGraph representing hardware connectivity
        gate_layers: Pre-computed layers of operations that can execute in parallel
        worst_spacing: Worst-case timesteps for token swapping (n-1 where n = number of qubits)
    """

    def __init__(
        self,
        coupling_map: nx.DiGraph,
        quantum_circuit: quariadne.circuit.AbstractQuantumCircuit,
    ):
        """Initialise the router with common setup.

        Args:
            quantum_circuit: AbstractQuantumCircuit containing operations to route
            coupling_map: NetworkX DiGraph representing physical qubit connectivity
        """
        self.circuit = quantum_circuit
        self.coupling_map = coupling_map

        # Extract operations and qubits
        self.two_qubit_operations = quantum_circuit.get_two_qubit_operations()
        self.qubits = quantum_circuit.qubits

        # Create gate layers for parallel operation execution
        # Maximum matching constraint ensures layer size respects hardware parallelism limits
        self.gate_layers = create_gate_layers(self.two_qubit_operations, coupling_map)

        # Calculate worst spacing for token swapping (n-1 where n = number of qubits)
        self.worst_spacing = len(quantum_circuit.qubits) - 1

    def _build_mapping_from_matrix(
        self, matrix: np.ndarray
    ) -> dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]:
        """Build a logical-to-physical mapping dictionary from a permutation/mapping matrix.

        Converts a 2D binary matrix (permutation or mapping matrix) where matrix[physical_idx, logical_idx] = 1
        indicates logical_idx is mapped to physical_idx.

        Args:
            matrix: 2D numpy array representing a permutation or mapping matrix

        Returns:
            Dictionary mapping LogicalQubit objects to PhysicalQubit objects.
        """
        mapping_dict = {}
        for physical_idx, logical_idx in np.argwhere(matrix):
            physical_qubit = quariadne.circuit.PhysicalQubit(physical_idx)
            logical_qubit = quariadne.circuit.LogicalQubit(logical_idx)
            mapping_dict[logical_qubit] = physical_qubit
        return mapping_dict

    @abstractmethod
    def get_initial_mapping(
        self,
    ) -> dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]:
        """Get initial mapping from logical to physical qubits.

        Returns:
            Dictionary mapping LogicalQubit objects to PhysicalQubit objects
            representing the initial qubit mapping.
        """
        pass

    @abstractmethod
    def get_inserted_swaps(self) -> dict[int, list[quariadne.circuit.PhysicalSwap]]:
        """Get swap operations inserted after each operation or layer.

        Returns:
            Dictionary mapping operation/layer indices to lists of PhysicalSwap objects.
            Each swap list contains swaps needed after executing the operation/layer at that index.
        """
        pass


class IlpRouter(Router):
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
        # Initialize base class
        super().__init__(coupling_map, quantum_circuit)

        # Run MILP optimisation
        milp_router = MilpScipyRouter(
            coupling_map=self.coupling_map,
            gate_layers=self.gate_layers,
            qubits=self.qubits,
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
        # Extract the mapping matrix at timestep 0 and convert to dictionary
        initial_timestep_mapping = self.result.mapping_variables[0]
        return self._build_mapping_from_matrix(initial_timestep_mapping)

    def get_inserted_swaps(self) -> dict[int, list[quariadne.circuit.PhysicalSwap]]:
        """Extract swap operations from qubit movement variables.

        TODO: Implement this operation e2e, now this is too raw.
        Analyses the qubit movement variables to identify SWAP operations that occur
        after each layer timestep. SWAP operations are detected by finding pairs of
        qubits that exchange positions between physical locations.

        Returns:
            Dictionary mapping layer indices (0-based) to lists of PhysicalSwap objects.
            Each swap pair is represented as a PhysicalSwap containing two PhysicalQubit objects.

        Example:
            {
                0: [],  # No swaps after layer 0
                1: [PhysicalSwap(PhysicalQubit(0), PhysicalQubit(1))],  # SWAP between physical qubits 0 and 1 after layer 1
                2: [PhysicalSwap(PhysicalQubit(2), PhysicalQubit(3)), PhysicalSwap(PhysicalQubit(0), PhysicalQubit(4))]  # Two SWAPs after layer 2
            }
        """
        swap_pairs_by_layer: defaultdict = defaultdict(list)

        # Find all non-zero movement variables where actual movement occurs
        defined_movements = np.argwhere(self.result.qubit_movement_variables)

        for movement in defined_movements:
            timestep, logical_qubit, from_qubit, to_qubit = movement.tolist()

            # Only consider actual movements (not self-assignments)
            if from_qubit != to_qubit:
                # Convert timestep to layer index using worst_spacing division
                # The +1 is an offset accounting for the fact that swaps happen AFTER respective layers
                layer_index = timestep // self.result.worst_spacing + 1

                # Create PhysicalSwap with proper qubit objects
                physical_qubit_from = quariadne.circuit.PhysicalQubit(from_qubit)
                physical_qubit_to = quariadne.circuit.PhysicalQubit(to_qubit)
                swap_pair = quariadne.circuit.PhysicalSwap(
                    physical_qubit_from, physical_qubit_to
                )

                # Get current layer's swap list
                current_layer_swaps = swap_pairs_by_layer[layer_index]

                # Add swap pair if not already present (avoids duplicates through PhysicalSwap equality)
                if swap_pair not in current_layer_swaps:
                    current_layer_swaps.append(swap_pair)

        return swap_pairs_by_layer


class LpRouterMapping(Router):
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
        # Initialize base class
        super().__init__(coupling_map, quantum_circuit)

        # Additional attributes for this router
        self.layer_count = len(self.gate_layers)

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
        mapping through optimisation. Extracts a compatible permutation using Birkhoff decomposition
        for the first gate layer.

        Returns:
            Initial permutation matrix
        """
        # Get first gate layer
        first_layer = self.gate_layers[0]

        # Run LP optimisation with continuous variables and no fixed mapping
        lp_router = MilpScipyRouter(
            coupling_map=self.coupling_map,
            gate_layers=self.gate_layers,
            qubits=self.qubits,
            integrality=CONTINUOUS_VARIABLE_INTEGRALITY,
            fixed_mapping=None,
        )
        lp_result = lp_router.run()

        # Extract compatible permutation for first gate layer
        bootstrap_mapping_variables = lp_result.mapping_variables[0]
        initial_permutation = self._extract_permutation_for_layer(
            bootstrap_mapping_variables, first_layer
        )

        return initial_permutation

    def _get_next_permutation(
        self,
        current_permutation: np.ndarray,
        current_layer: GateLayer,
        remaining_layers: list[GateLayer],
    ) -> np.ndarray:
        """Get next permutation by running LP on remaining layers with fixed current permutation.

        Args:
            current_permutation: Current permutation matrix to fix as constraint
            current_layer: Gate layer that needs to be made executable
            remaining_layers: List of remaining gate layers to route

        Returns:
            Next permutation matrix compatible with all operations in the current layer
        """
        # Run LP on remaining layers with current permutation fixed
        fixed_permutation_array = current_permutation[np.newaxis, ...]

        lp_router = MilpScipyRouter(
            coupling_map=self.coupling_map,
            gate_layers=remaining_layers,
            qubits=self.qubits,
            integrality=CONTINUOUS_VARIABLE_INTEGRALITY,
            fixed_mapping=fixed_permutation_array,
        )
        lp_result = lp_router.run()

        # Extract next permutation at second layer's timestep (first layer at timestep 0, second at worst_spacing)
        next_mapping_timestep = lp_result.worst_spacing
        next_mapping_variables = lp_result.mapping_variables[next_mapping_timestep]

        next_permutation = self._extract_permutation_for_layer(
            next_mapping_variables, current_layer
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

    def _is_layer_compatible_with_permutation(
        self,
        permutation_matrix: np.ndarray,
        layer: GateLayer,
    ) -> bool:
        """Check if a permutation matrix is compatible with ALL operations in a gate layer.

        Args:
            permutation_matrix: 2D numpy array representing a permutation matrix
            layer: Gate layer containing operations that need to execute in parallel

        Returns:
            True if the permutation allows ALL operations in the layer to be connected, False otherwise.
        """
        # Create the mapping from physical to logical qubits for this permutation
        physical_to_logical_mapping = self._create_physical_to_logical_mapping(
            permutation_matrix
        )
        # Relabel the coupling map to get logical connectivity
        logical_connectivity = nx.relabel_nodes(
            self.coupling_map, physical_to_logical_mapping
        )

        # Check if ALL required edges for the layer exist in logical connectivity
        return all(
            logical_connectivity.has_edge(left_logical, right_logical)
            for operation in layer
            for left_logical, right_logical in [operation.qubits_participating]
        )

    def _build_mapping_from_permutation(
        self, permutation_matrix: np.ndarray
    ) -> dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]:
        """Build a logical-to-physical mapping dictionary from a permutation matrix.

        Args:
            permutation_matrix: 2D numpy array representing a permutation matrix

        Returns:
            Dictionary mapping LogicalQubit objects to PhysicalQubit objects.
        """
        return self._build_mapping_from_matrix(permutation_matrix)

    def _extract_permutation_for_layer(
        self,
        mapping_variables: np.ndarray,
        layer: GateLayer,
    ) -> np.ndarray:
        """Extract permutation matrix for a gate layer using Birkhoff decomposition.

        Uses Birkhoff-von Neumann decomposition to convert doubly stochastic matrix to
        permutations and selects the first permutation that satisfies connectivity
        requirements for ALL operations in the layer.

        Args:
            mapping_variables: 2D numpy array representing doubly stochastic mapping matrix
            layer: Gate layer containing operations that need to execute in parallel

        Returns:
            2D numpy array representing the permutation matrix.

        Raises:
            ValueError: If no compatible permutation found for the layer.
        """
        # Perform Birkhoff-von Neumann decomposition
        coefficient_matrix_pairs = birkhoff_von_neumann_decomposition(mapping_variables)

        # Sort permutations by coefficient in descending order
        sorted_permutations = sorted(
            coefficient_matrix_pairs,
            key=lambda coefficient_matrix_pair: coefficient_matrix_pair[0],
            reverse=True,
        )

        # Find the first permutation that satisfies connectivity requirements for ALL operations in layer
        for coefficient, permutation_matrix in sorted_permutations:
            if self._is_layer_compatible_with_permutation(permutation_matrix, layer):
                return permutation_matrix

        # If no compatible permutation found, raise an error
        raise ValueError(f"No compatible permutation found for layer {layer}")

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

        # Extract indices using keys() and values() which maintain correspondence
        logical_indices = [logical_qubit.index for logical_qubit in mapping.keys()]
        physical_indices = [physical_qubit.index for physical_qubit in mapping.values()]

        # Use advanced indexing to assign all values at once
        permutation_matrix[physical_indices, logical_indices] = 1.0

        return permutation_matrix

    def _get_swaps_between_two_permutations(
        self,
        previous_permutation: np.ndarray,
        current_permutation: np.ndarray,
        previous_layer: GateLayer,
        current_layer: GateLayer,
    ) -> list[quariadne.circuit.PhysicalSwap]:
        """Calculate swap operations between two consecutive permutations.

        Runs a constrained routing problem with exactly two gate layers and two fixed
        permutations to recover the minimal sequence of swaps needed to transition
        between them.

        Args:
            previous_permutation: Permutation matrix for the previous layer
            current_permutation: Permutation matrix for the current layer
            previous_layer: The previous gate layer
            current_layer: The current gate layer

        Returns:
            List of PhysicalSwap operations needed to transition between permutations
        """
        two_step_permutations = np.array([previous_permutation, current_permutation])
        two_step_layers = [previous_layer, current_layer]

        # Run router with both layers and fixed mappings
        lp_router = MilpScipyRouter(
            coupling_map=self.coupling_map,
            gate_layers=two_step_layers,
            qubits=self.qubits,
            integrality=CONTINUOUS_VARIABLE_INTEGRALITY,
            fixed_mapping=two_step_permutations,
        )
        lp_result = lp_router.run()

        # Phase 1: Accumulate edge swaps
        swap_edges = accumulate_edge_swaps(lp_result.qubit_movement_variables)

        # Phase 2: Create list of swaps
        swaps = create_swaps_from_edges(swap_edges)

        return swaps

    def _run(self) -> None:
        """Run the iterative LP routing process to generate sequence of mappings and swaps.

        Implements the iterative routing algorithm:
        1. Bootstrap: get initial permutation using LP and Birkhoff decomposition for first layer
        2. Iterate through gate layers:
           - Run LP on remaining layers to get next permutation
           - Calculate swaps between previous and current permutation
           - Pop the processed layer
        """
        # Make a working copy of layers list
        remaining_layers = self.gate_layers.copy()

        # Bootstrap phase: get initial permutation
        current_permutation = self._get_initial_permutation()
        initial_mapping = self._build_mapping_from_permutation(current_permutation)
        self.mappings = [initial_mapping]
        self.permutations = [current_permutation]
        # Iterative phase: process layers one by one
        for layer_index in range(1, self.layer_count):
            routed_layer = self.gate_layers[layer_index]
            previous_layer = self.gate_layers[layer_index - 1]
            previous_permutation = current_permutation
            # Get next permutation for this layer
            next_permutation = self._get_next_permutation(
                current_permutation, routed_layer, remaining_layers
            )
            next_mapping = self._build_mapping_from_permutation(next_permutation)

            # Calculate swaps between previous and next permutation
            swaps = self._get_swaps_between_two_permutations(
                previous_permutation,
                next_permutation,
                previous_layer,
                routed_layer,
            )
            self.swaps_by_operation[layer_index] = swaps

            # Update current permutation and add mapping to list
            self.mappings.append(next_mapping)
            self.permutations.append(next_permutation)
            current_permutation = next_permutation

            # Pop the first layer from remaining layers
            remaining_layers.pop(0)

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


class LpRouterEdges(Router):
    """Linear Programming router using edge-based assignment with bipartite matching.

    This router uses a single LP run and then applies bipartite matching to extract
    edge assignments. The process:
    1. Run LP with continuous variables on all layers
    2. For each layer, aggregate gate execution weights per edge
    3. Build bipartite graph via node splitting
    4. Find maximum weight matching
    5. Greedily assign operations to matched edges

    Attributes:
        mappings: List of extracted mappings per layer
        swaps_by_layer: Dictionary mapping layer indices to swap lists
        edge_assignments: Edge assignment array from bipartite matching
    """

    def __init__(
        self,
        coupling_map: nx.DiGraph,
        quantum_circuit: quariadne.circuit.AbstractQuantumCircuit,
    ):
        """Initialize the edge-based LP router and automatically run the routing.

        Args:
            coupling_map: NetworkX DiGraph representing the physical qubit connectivity
            quantum_circuit: Abstract quantum circuit representation to be routed
        """
        # Initialize base class
        super().__init__(coupling_map, quantum_circuit)

        # Additional attributes for this router
        self.layer_count = len(self.gate_layers)
        self.mappings: list[
            dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]
        ] = []
        self.swaps_by_layer: dict[int, list[quariadne.circuit.PhysicalSwap]] = {}
        self.edge_assignments: dict[GateLayer, list[OperationEdgeAssignment]] | None = (
            None
        )

        # Pre-compute edge-to-index lookup dictionary for O(1) lookups
        # Replaces O(n) list.index() calls in bipartite matching translation
        self._edge_to_index: dict[
            tuple[quariadne.circuit.PhysicalQubit, quariadne.circuit.PhysicalQubit], int
        ] = {edge: idx for idx, edge in enumerate(self.coupling_map.edges())}

        # Run routing automatically
        self._run()

    def _split_coupling_map_to_bipartite(
        self, layer_execution_weights: np.ndarray
    ) -> tuple[nx.Graph, dict[str, quariadne.circuit.PhysicalQubit]]:
        """Split coupling map nodes to create bipartite graph for matching.

        Each physical qubit v becomes two nodes: v_in and v_out.
        Directed edge u -> v becomes undirected edge u_out <-> v_in.
        Forcing edges v_in <-> v_out ensure all nodes are matched.

        Args:
            layer_execution_weights: Aggregated gate execution weight for each edge
                                     in coupling map, shape (edge_count,)

        Returns:
            Tuple of (bipartite_graph, node_to_qubit_mapping) where:
            - bipartite_graph: Graph with node-split structure
            - node_to_qubit_mapping: Dict mapping bipartite node names (e.g.,
              "PhysicalQubit(index=0)_in") to their corresponding PhysicalQubit objects

        See Also:
            NetworkX bipartite module:
            https://networkx.org/documentation/stable/reference/algorithms/bipartite.html
        """
        bipartite_coupling_map: nx.Graph = nx.Graph()
        node_to_qubit: dict[str, quariadne.circuit.PhysicalQubit] = {}

        for physical_qubit in self.coupling_map.nodes:
            physical_qubit_in = f"{physical_qubit}_in"
            physical_qubit_out = f"{physical_qubit}_out"

            # Store reverse mapping from bipartite node names to physical qubits
            node_to_qubit[physical_qubit_in] = physical_qubit
            node_to_qubit[physical_qubit_out] = physical_qubit

            bipartite_coupling_map.add_node(physical_qubit_in, bipartite=0)
            bipartite_coupling_map.add_node(physical_qubit_out, bipartite=1)

            bipartite_coupling_map.add_edge(
                physical_qubit_in, physical_qubit_out, weight=BIPARTITE_FORCING_TERM
            )

        for edge_idx, (source_qubit, target_qubit) in enumerate(
            self.coupling_map.edges()
        ):
            execution_weight = layer_execution_weights[edge_idx]
            source_qubit_out = f"{source_qubit}_out"
            target_qubit_in = f"{target_qubit}_in"
            bipartite_coupling_map.add_edge(
                source_qubit_out,
                target_qubit_in,
                weight=execution_weight + BIPARTITE_FORCING_TERM,
            )

        return bipartite_coupling_map, node_to_qubit

    def _extract_bipartite_partitions(
        self, bipartite_graph: nx.Graph
    ) -> tuple[list[str], list[str]]:
        """Extract partitions from bipartite graph using bipartite node attribute.

        Separates nodes into two partitions based on their bipartite attribute value.
        Partition ordering is stable based on graph node order.

        Args:
            bipartite_graph: Bipartite graph with bipartite attribute on nodes

        Returns:
            Tuple of (first_partition, second_partition) corresponding to
            bipartite attribute values 0 and 1 respectively
        """
        first_partition = [
            node
            for node, data in bipartite_graph.nodes(data=True)
            if data["bipartite"] == 0
        ]
        second_partition = [
            node
            for node, data in bipartite_graph.nodes(data=True)
            if data["bipartite"] == 1
        ]
        return first_partition, second_partition

    def _translate_bipartite_matching_to_edges(
        self,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
        out_partition: list[str],
        in_partition: list[str],
        node_to_qubit: dict[str, quariadne.circuit.PhysicalQubit],
    ) -> list[int]:
        """Translate bipartite matching result to coupling map edge indices.

        The bipartite matching returns indices into the partitions:
        - row_indices: indices into out_partition (qubit_out nodes)
        - col_indices: indices into in_partition (qubit_in nodes)

        Uses explicit node_to_qubit mapping to correctly resolve physical qubits
        from partition node names, avoiding any assumptions about partition ordering.

        The matching result contains two types of edges:
        1. Forcing edges: where source_qubit == target_qubit, representing
           qubit_in <-> qubit_out for the same qubit (qubit not in any gate).
        2. Actual edges: where source_qubit != target_qubit, representing
           source_out <-> target_in, which maps to edge source -> target.

        Args:
            row_indices: Row indices from matching, indexing into out_partition
            col_indices: Column indices from matching, indexing into in_partition
            out_partition: List of bipartite node names for the _out partition
            in_partition: List of bipartite node names for the _in partition
            node_to_qubit: Mapping from bipartite node names to PhysicalQubit objects

        Returns:
            List of edge indices into self.coupling_map.edges() for matched actual edges

        See Also:
            scipy.sparse.csgraph.min_weight_full_bipartite_matching:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.min_weight_full_bipartite_matching.html
        """
        matched_edge_indices = []

        for row_idx, col_idx in zip(row_indices, col_indices):
            # Resolve physical qubits via explicit mapping (not index assumption)
            source_node = out_partition[row_idx]
            target_node = in_partition[col_idx]

            source_qubit = node_to_qubit[source_node]
            target_qubit = node_to_qubit[target_node]

            # Skip forcing edges (self-loops indicate qubit not used in gate)
            if source_qubit != target_qubit:
                edge = (source_qubit, target_qubit)
                # O(1) dictionary lookup instead of O(n) list.index()
                if edge in self._edge_to_index:
                    edge_index = self._edge_to_index[edge]
                    matched_edge_indices.append(edge_index)

        return matched_edge_indices

    def _assign_operations_to_edges_greedily(
        self,
        layer_gate_execution: np.ndarray,
        matched_edge_indices: list[int],
    ) -> list[OperationEdgeAssignment]:
        """Greedily assign operations to matched edges by highest execution value.

        For each operation in the layer, selects the matched edge with the highest
        gate execution value that hasn't been assigned yet.

        Args:
            layer_gate_execution: Gate execution values for this layer,
                                  shape (operations_in_layer, edge_count)
            matched_edge_indices: List of edge indices selected by bipartite matching

        Returns:
            List of (local_operation_index, edge_index) tuples representing assignments
        """
        assignments = []
        available_edges = list(matched_edge_indices)
        operations_in_layer = layer_gate_execution.shape[0]

        for local_op_idx in range(operations_in_layer):
            edge_values = layer_gate_execution[local_op_idx, available_edges]

            best_idx_in_available = int(np.argmax(edge_values))
            best_edge_idx = available_edges[best_idx_in_available]

            assignments.append((local_op_idx, best_edge_idx))
            available_edges.pop(best_idx_in_available)

        return assignments

    def _prune_matched_edges_by_vertex_usage(
        self,
        matched_edge_indices: list[int],
        layer_execution_weights: np.ndarray,
    ) -> list[int]:
        """Prune matched edges to ensure each vertex is used at most once.

        After bipartite matching, vertices may be reused across multiple edges,
        violating perfect matching constraints. This function sorts edges by
        their aggregated execution weight and greedily selects edges, ensuring
        each physical qubit appears in at most one edge.

        Args:
            matched_edge_indices: List of edge indices selected by bipartite matching
            layer_execution_weights: Aggregated gate execution weight for each edge
                                     in coupling map, shape (edge_count,)

        Returns:
            Pruned list of edge indices where each vertex appears at most once,
            prioritised by execution weight (highest weight edges kept first)
        """
        coupling_map_edges = list(self.coupling_map.edges())

        # Build list of edges with their execution weights for sorting
        weighted_edges = list(
            (*coupling_map_edges[edge_idx], layer_execution_weights[edge_idx])
            for edge_idx in matched_edge_indices
        )

        # Sort edges by weight in descending order (highest weight first)
        sorted_edges = sorted(
            weighted_edges,
            key=lambda edge_with_weight: edge_with_weight[
                2
            ],  # edge_with_weight[2] is the weight
            reverse=True,
        )

        # Greedily select edges, tracking used vertices
        pruned_edge_indices = []
        used_vertices = set()

        for source_qubit, target_qubit, weight in sorted_edges:
            # Only keep edge if neither vertex has been used
            if source_qubit not in used_vertices and target_qubit not in used_vertices:
                # O(1) dictionary lookup instead of O(n) list.index()
                edge_idx = self._edge_to_index[(source_qubit, target_qubit)]
                pruned_edge_indices.append(edge_idx)
                used_vertices.add(source_qubit)
                used_vertices.add(target_qubit)

        return pruned_edge_indices

    def _generate_initial_mapping_from_edges(
        self,
    ) -> dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]:
        """Generate initial mapping from first layer's edge assignments.

        Uses a two-phase approach:
        1. Map logical qubits involved in operations to their assigned physical edges
        2. Map remaining logical qubits trivially to unused physical qubits

        Returns:
            Dictionary mapping LogicalQubit to PhysicalQubit for initial layout
        """
        initial_mapping: dict[
            quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit
        ] = {}

        # Get complete sets of all qubits
        all_logical_qubits = set(self.qubits)
        all_physical_qubits = set(self.coupling_map.nodes)

        # Phase 1: Map qubits constrained by edge assignments
        if self.edge_assignments is None:
            raise RuntimeError(
                "edge_assignments must be set before generating initial mapping"
            )
        first_layer = self.gate_layers[0]
        first_layer_assignments = self.edge_assignments[first_layer]
        coupling_map_edges = list(self.coupling_map.edges())

        mapped_logical_qubits = set()
        mapped_physical_qubits = set()

        for operation_local_index, edge_index in first_layer_assignments:
            operation = first_layer[operation_local_index]
            edge_source_qubit, edge_target_qubit = coupling_map_edges[edge_index]
            operation_left_qubit, operation_right_qubit = operation.qubits_participating

            # Map logical qubits to physical edge qubits
            initial_mapping[operation_left_qubit] = edge_source_qubit
            initial_mapping[operation_right_qubit] = edge_target_qubit

            # Track mapped qubits
            mapped_logical_qubits.add(operation_left_qubit)
            mapped_logical_qubits.add(operation_right_qubit)
            mapped_physical_qubits.add(edge_source_qubit)
            mapped_physical_qubits.add(edge_target_qubit)

        # Remove mapped qubits in batch
        remaining_logical_qubits = all_logical_qubits - mapped_logical_qubits
        remaining_physical_qubits = all_physical_qubits - mapped_physical_qubits

        # Phase 2: Map remaining qubits deterministically (sorted by index)
        # Sorting ensures reproducible results across runs
        # See: https://docs.python.org/3/library/stdtypes.html#set (sets are unordered)
        remaining_logical_sorted = sorted(
            remaining_logical_qubits, key=lambda q: q.index
        )
        remaining_physical_sorted = sorted(
            remaining_physical_qubits, key=lambda q: q.index
        )

        for logical_qubit, physical_qubit in zip(
            remaining_logical_sorted, remaining_physical_sorted
        ):
            initial_mapping[logical_qubit] = physical_qubit

        # Validate that the initial mapping respects coupling constraints
        self._validate_initial_mapping(initial_mapping, first_layer)

        return initial_mapping

    def _validate_initial_mapping(
        self,
        mapping: dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit],
        first_layer: GateLayer,
    ) -> None:
        """Validate that initial mapping respects coupling map constraints.

        Checks that for each two-qubit operation in the first layer, the assigned
        physical qubits form a valid edge in the coupling map.

        Args:
            mapping: The initial logical-to-physical qubit mapping
            first_layer: The first gate layer containing operations to validate

        Raises:
            ValueError: If any operation's physical qubits are not adjacent in
                the coupling map

        See Also:
            Qiskit coupling map documentation:
            https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.CouplingMap
        """
        for operation in first_layer:
            left_logical, right_logical = operation.qubits_participating
            left_physical = mapping[left_logical]
            right_physical = mapping[right_logical]

            edge = (left_physical, right_physical)
            reverse_edge = (right_physical, left_physical)

            has_forward_edge = edge in self._edge_to_index
            has_reverse_edge = reverse_edge in self._edge_to_index

            if not has_forward_edge and not has_reverse_edge:
                raise ValueError(
                    f"Initial mapping violates coupling constraint: "
                    f"operation {operation} maps to physical qubits "
                    f"{left_physical} and {right_physical}, but neither edge "
                    f"{edge} nor {reverse_edge} exists in coupling map"
                )

    def get_initial_mapping(
        self,
    ) -> dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]:
        """Get initial mapping from the first mapping in mappings list.

        Returns:
            Dictionary mapping LogicalQubit objects to PhysicalQubit objects
            representing the initial qubit mapping.
        """
        return self.mappings[0]

    def get_inserted_swaps(self) -> dict[int, list[quariadne.circuit.PhysicalSwap]]:
        """Get swap operations inserted after each layer.

        Returns:
            Dictionary mapping layer indices to lists of PhysicalSwap objects.
            Each swap list contains swaps needed after executing the layer at that index.
        """
        return self.swaps_by_layer

    def _extract_edge_assignments_with_bipartite_matching(
        self, gate_execution_variables: np.ndarray
    ) -> dict[GateLayer, list[OperationEdgeAssignment]]:
        """Extract edge assignments using bipartite matching on node-split graph.

        For each layer:
        1. Aggregate gate execution weights per edge
        2. Build bipartite graph via node splitting
        3. Find maximum weight matching
        4. Greedily assign operations to matched edges

        Args:
            gate_execution_variables: Gate execution variables from LP solution
                                     Shape: (operation_count, edge_count)

        Returns:
            Dictionary mapping each GateLayer to its list of (operation_index, edge_index) assignments.
            Operation indices are local to each layer.
        """
        # Build dictionary keyed by GateLayer
        edge_assignments_dict: dict[GateLayer, list[OperationEdgeAssignment]] = {}

        # Track starting operation index for each layer
        starting_operation_index = 0

        for layer_idx in range(self.layer_count):
            # Get the gate layer object to use as dictionary key
            gate_layer: GateLayer = self.gate_layers[layer_idx]
            operations_in_layer = len(gate_layer)

            # Calculate ending index (exclusive)
            ending_operation_index = starting_operation_index + operations_in_layer

            # Extract gate execution matrix for this layer using global operation indices
            layer_gate_execution = gate_execution_variables[
                starting_operation_index:ending_operation_index, :
            ]

            # Aggregate weights per edge across all operations in layer
            layer_execution_weights = np.sum(layer_gate_execution, axis=0)

            # Build bipartite graph with node-to-qubit mapping for correct translation
            bipartite_coupling_map, node_to_qubit = (
                self._split_coupling_map_to_bipartite(layer_execution_weights)
            )

            # Extract partitions for biadjacency matrix construction
            in_partition, out_partition = self._extract_bipartite_partitions(
                bipartite_coupling_map
            )

            biadjacency_matrix = nx.bipartite.biadjacency_matrix(
                bipartite_coupling_map, out_partition, in_partition
            )

            row_indices, col_indices = (
                scipy.sparse.csgraph.min_weight_full_bipartite_matching(
                    biadjacency_matrix, maximize=True
                )
            )

            # Translate matching to coupling map edge indices using explicit mapping
            matched_edge_indices = self._translate_bipartite_matching_to_edges(
                row_indices,
                col_indices,
                list(out_partition),
                list(in_partition),
                node_to_qubit,
            )

            # Prune edges to ensure each vertex appears at most once
            pruned_edge_indices = self._prune_matched_edges_by_vertex_usage(
                matched_edge_indices, layer_execution_weights
            )

            # Greedily assign operations to matched edges
            operation_edge_assignments = self._assign_operations_to_edges_greedily(
                layer_gate_execution, pruned_edge_indices
            )

            # Store assignments in dictionary using GateLayer as key
            # Operation indices are already local to this layer from the greedy assignment
            edge_assignments_dict[gate_layer] = operation_edge_assignments

            # Update starting index for next layer
            starting_operation_index = ending_operation_index

        return edge_assignments_dict

    def _get_swaps_between_two_edge_fixed_layers(
        self,
        layer_1: GateLayer,
        layer_2: GateLayer,
        layer_1_assignments: list[OperationEdgeAssignment],
        layer_2_assignments: list[OperationEdgeAssignment],
    ) -> list[quariadne.circuit.PhysicalSwap]:
        """Calculate swap operations between two consecutive edge-fixed layers.

        Runs a constrained routing problem with exactly two gate layers and fixed
        edge assignments to recover the minimal sequence of swaps needed to transition
        between them.

        Args:
            layer_1: First gate layer
            layer_2: Second gate layer
            layer_1_assignments: List of (operation_index, edge_index) assignments for first layer
            layer_2_assignments: List of (operation_index, edge_index) assignments for second layer

        Returns:
            List of PhysicalSwap operations needed between the layers
        """
        # Package edge assignments as list for fixed_edges parameter
        two_step_edges = [layer_1_assignments, layer_2_assignments]
        two_step_layers = [layer_1, layer_2]

        # Run router with both layers and fixed edge assignments
        lp_router = MilpScipyRouter(
            coupling_map=self.coupling_map,
            gate_layers=two_step_layers,
            qubits=self.qubits,
            integrality=CONTINUOUS_VARIABLE_INTEGRALITY,
            fixed_edges=two_step_edges,
        )
        lp_result = lp_router.run()

        # Phase 1: Accumulate edge swaps
        swap_edges = accumulate_edge_swaps(lp_result.qubit_movement_variables)

        # Phase 2: Create list of swaps
        swaps = create_swaps_from_edges(swap_edges)

        return swaps

    def _run(self) -> None:
        """Run the edge-based LP routing process.

        Algorithm:
        1. Run LP with continuous variables on all layers
        2. Extract edge assignments using bipartite matching
        3. For each consecutive pair of layers, run small LP with fixed edges to recover swaps
        """
        # Phase 1: Initial LP run with continuous variables
        initial_lp_router = MilpScipyRouter(
            coupling_map=self.coupling_map,
            gate_layers=self.gate_layers,
            qubits=self.qubits,
            integrality=CONTINUOUS_VARIABLE_INTEGRALITY,
            fixed_mapping=None,
        )
        initial_lp_result = initial_lp_router.run()

        # Phase 2: Extract edge assignments using bipartite matching
        self.edge_assignments = self._extract_edge_assignments_with_bipartite_matching(
            initial_lp_result.gate_execution_variables
        )

        # Phase 2.5: Generate initial mapping from first layer edge assignments
        initial_mapping = self._generate_initial_mapping_from_edges()
        self.mappings = [initial_mapping]

        # Phase 3: Iteratively recover swaps between consecutive layers
        for layer_idx in range(self.layer_count - 1):
            current_layer = self.gate_layers[layer_idx]
            next_layer = self.gate_layers[layer_idx + 1]

            # Extract edge assignments from dictionary using GateLayer keys
            current_layer_assignments = self.edge_assignments[current_layer]
            next_layer_assignments = self.edge_assignments[next_layer]

            swaps = self._get_swaps_between_two_edge_fixed_layers(
                current_layer,
                next_layer,
                current_layer_assignments,
                next_layer_assignments,
            )
            self.swaps_by_layer[layer_idx] = swaps
