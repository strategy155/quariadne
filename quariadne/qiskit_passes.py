from typing import List, Dict, Union
from enum import Enum
import quariadne.computational_graph
import quariadne.milp
import quariadne.circuit
import qiskit.transpiler
import qiskit.dagcircuit
import qiskit.circuit.library
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.preset_passmanagers.common import generate_embed_passmanager
from qiskit.transpiler import PassManager

import quariadne.routers
import quariadne.bipartite_allocation_lp


class RoutingMode(Enum):
    """Routing mode for quantum circuit routing optimisation.

    Attributes:
        ILP: Integer Linear Program with integer constraints
        LP: Linear Program with continuous variables and Birkhoff decomposition
        EDGE: Linear Program with edge-based bipartite matching
    """

    ILP = "ilp"
    LP = "lp"
    EDGE = "edge"


class MilpLayout(qiskit.transpiler.AnalysisPass):
    """Layout pass using MILP optimisation for initial qubit placement."""

    def __init__(
        self,
        coupling_map: Union[qiskit.transpiler.CouplingMap, qiskit.transpiler.Target],
        mode: RoutingMode = RoutingMode.ILP,
    ) -> None:
        """Initialise MILP layout pass with backend coupling constraints.

        Args:
            coupling_map: Backend coupling map defining physical qubit connectivity
            mode: Routing mode (RoutingMode.ILP for integer program or RoutingMode.LP for linear program)
        """
        super().__init__()
        self.mode = mode

        if isinstance(coupling_map, qiskit.transpiler.Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map

        self.coupling_graph = quariadne.milp.get_coupling_graph(self.coupling_map)

    def _convert_dag_to_circuit(
        self, dag: qiskit.dagcircuit.DAGCircuit
    ) -> quariadne.circuit.AbstractQuantumCircuit:
        """Convert Qiskit DAG to Quariadne abstract quantum circuit representation.

        Args:
            dag: Input Qiskit DAG circuit to convert

        Returns:
            Quariadne abstract quantum circuit for MILP optimization
        """
        # Transform DAG through computational graph intermediate representation
        quariadne_dag = quariadne.computational_graph.ComputationalDAG.from_qiskit_dag(
            dag
        )
        return quariadne_dag.to_abstract_quantum_circuit()

    def _generate_physical_qubit_indices(
        self,
        physical_by_logical_mapping: Dict[
            quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit
        ],
        dag_qubit_count: int,
    ) -> List[int]:
        """Extract physical qubit indices from MILP mapping solution for layout creation.

        Args:
            physical_by_logical_mapping: MILP solution mapping logical to physical qubits
            dag_qubit_count: Number of qubits in the original DAG circuit

        Returns:
            List of physical qubit indices ordered by logical qubit index
        """
        # Sort logical qubits by index to maintain consistent ordering
        sorted_logical_qubits = sorted(
            physical_by_logical_mapping.keys(), key=lambda qubit: qubit.index
        )

        # Filter to only include qubits that exist in the DAG, excluding dummy qubits
        # TODO: DIRTY THING DUE TO THE WAY HOW QISKIT WORKS
        valid_logical_qubits = [
            qubit for qubit in sorted_logical_qubits if qubit.index < dag_qubit_count
        ]

        # Extract corresponding physical qubit indices for layout construction
        physical_qubit_indices = []
        for logical_qubit in valid_logical_qubits:
            physical_qubit = physical_by_logical_mapping[logical_qubit]
            physical_qubit_indices.append(physical_qubit.index)

        return physical_qubit_indices

    def run(self, dag: qiskit.dagcircuit.DAGCircuit) -> None:
        """Execute ILP/LP optimisation to determine optimal initial qubit layout."""
        if self.target is not None:
            if dag.num_qubits() > self.target.num_qubits:
                raise qiskit.transpiler.TranspilerError(
                    "Number of qubits greater than device."
                )
        elif dag.num_qubits() > self.coupling_map.size():
            raise qiskit.transpiler.TranspilerError(
                "Number of qubits greater than device."
            )

        # Convert to internal representation and solve using appropriate router
        quariadne_circuit = self._convert_dag_to_circuit(dag)

        router: quariadne.routers.Router
        if self.mode == RoutingMode.ILP:
            router = quariadne.routers.IlpRouter(self.coupling_graph, quariadne_circuit)
        elif self.mode == RoutingMode.LP:
            router = quariadne.routers.LpRouterMapping(
                self.coupling_graph, quariadne_circuit
            )
        elif self.mode == RoutingMode.EDGE:
            router = quariadne.routers.LpRouterEdges(
                self.coupling_graph, quariadne_circuit
            )
        else:
            raise ValueError(f"Unknown routing mode: {self.mode}")

        # Store router object for potential use by routing pass
        self.property_set["router"] = router
        physical_by_logical_initial_mapping = router.get_initial_mapping()

        # Generate layout from optimisation solution
        physical_qubit_indices = self._generate_physical_qubit_indices(
            physical_by_logical_initial_mapping, dag.num_qubits()
        )

        # Generation of the final layout
        canonical_register = dag.qregs["q"]
        layout = qiskit.transpiler.Layout.from_intlist(
            physical_qubit_indices, canonical_register
        )
        self.property_set["layout"] = layout


class MilpRouting(qiskit.transpiler.TransformationPass):
    """Routing pass using MILP optimisation for SWAP insertion."""

    def __init__(self, coupling_map: qiskit.transpiler.CouplingMap) -> None:
        """Initialise MILP routing pass with backend coupling constraints.

        Args:
            coupling_map: Backend coupling map defining physical qubit connectivity
        """
        super().__init__()
        self.coupling_map = coupling_map

    def _create_swap_dag(
        self,
        canonical_register: qiskit.circuit.QuantumRegister,
        current_layout: qiskit.transpiler.Layout,
        swaps: List[quariadne.circuit.PhysicalSwap],
    ) -> qiskit.dagcircuit.DAGCircuit:
        """Create DAG containing SWAP operations for a specific layer.

        Args:
            canonical_register: Quantum register for the circuit
            current_layout: Current mapping of qubits to physical positions
            swaps: List of physical qubit pairs that need to be swapped

        Returns:
            DAG circuit containing all required SWAP operations
        """
        swap_dag = qiskit.dagcircuit.DAGCircuit()
        swap_dag.add_qreg(canonical_register)

        # Add each required SWAP operation to the DAG
        for swap in swaps:
            physical_qubit_1, physical_qubit_2 = swap.first.index, swap.second.index
            logical_qubit_1, logical_qubit_2 = (
                current_layout[physical_qubit_1],
                current_layout[physical_qubit_2],
            )

            swap_dag.apply_operation_back(
                qiskit.circuit.library.SwapGate(),
                (logical_qubit_1, logical_qubit_2),
                cargs=(),
                check=False,
            )

        return swap_dag

    def _apply_swaps_to_layout(
        self,
        current_layout: qiskit.transpiler.Layout,
        swaps: List[quariadne.circuit.PhysicalSwap],
    ) -> qiskit.transpiler.Layout:
        """Apply SWAP operations to layout and return updated layout.

        Args:
            current_layout: Current mapping of qubits to physical positions
            swaps: List of physical qubit pairs that need to be swapped

        Returns:
            New layout with SWAP operations applied
        """
        updated_layout = current_layout.copy()

        # Apply each SWAP operation to the layout
        for swap in swaps:
            physical_qubit_1, physical_qubit_2 = swap.first.index, swap.second.index
            updated_layout.swap(physical_qubit_1, physical_qubit_2)

        return updated_layout

    def run(self, dag: qiskit.dagcircuit.DAGCircuit) -> qiskit.dagcircuit.DAGCircuit:
        """Apply MILP-determined SWAP operations to the DAG."""
        new_dag = dag.copy_empty_like()

        if self.coupling_map is None:
            raise qiskit.transpiler.TranspilerError(
                "MilpRouting cannot run with coupling_map=None"
            )

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise qiskit.transpiler.TranspilerError(
                "MILP routing runs on physical circuits only"
            )

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise qiskit.transpiler.TranspilerError(
                "Layout does not match DAG qubit count"
            )

        # Retrieve router from property set (populated by MilpLayout)
        if "router" not in self.property_set:
            raise qiskit.transpiler.TranspilerError(
                "MilpRouting requires MilpLayout to be run first"
            )

        canonical_register = dag.qregs["q"]
        router = self.property_set["router"]
        swap_pairs_by_timestep = router.get_inserted_swaps()

        # Initialise layout tracking
        trivial_layout = qiskit.transpiler.Layout.generate_trivial_layout(
            canonical_register
        )
        current_layout = trivial_layout.copy()

        layer_idx = 0
        # Process each parallel layer and insert SWAPs after layers
        for layer in dag.layers():
            subdag = layer["graph"]

            # Add the entire parallel layer first
            order = current_layout.reorder_bits(new_dag.qubits)
            new_dag.compose(subdag, qubits=order)

            # Insert SWAPs after layer if they exist
            if layer_idx in swap_pairs_by_timestep:
                swaps = swap_pairs_by_timestep[layer_idx]
                # Create and insert SWAP operations
                swap_dag = self._create_swap_dag(
                    canonical_register, current_layout, swaps
                )
                order = current_layout.reorder_bits(new_dag.qubits)
                new_dag.compose(swap_dag, qubits=order)

                # Update layout state after SWAPs
                current_layout = self._apply_swaps_to_layout(current_layout, swaps)

            layer_idx += 1

        # Update final layout state in property set
        if self.property_set["final_layout"] is None:
            self.property_set["final_layout"] = current_layout
        else:
            self.property_set["final_layout"] = self.property_set[
                "final_layout"
            ].compose(current_layout, dag.qubits)

        return new_dag


class QuariadneIlpLayoutPlugin(PassManagerStagePlugin):
    """PassManager stage plugin for MILP-based layout optimization."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        """Generate PassManager for MILP layout stage.

        Args:
            pass_manager_config: Pass manager configuration object
            optimization_level: Optimization level (0-3)

        Returns:
            PassManager for MILP layout stage
        """
        layout_pm = PassManager()

        # Add MILP layout pass
        if (
            hasattr(pass_manager_config, "target")
            and pass_manager_config.target is not None
        ):
            layout_pm.append(MilpLayout(pass_manager_config.target))
        else:
            layout_pm.append(MilpLayout(pass_manager_config.coupling_map))

        # Embed the layout using generate_embed_passmanager
        embed_pm = generate_embed_passmanager(pass_manager_config.coupling_map)
        layout_pm += embed_pm

        return layout_pm


class QuariadneIlpRoutingPlugin(PassManagerStagePlugin):
    """PassManager stage plugin for MILP-based routing optimisation (ILP mode)."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        """Generate PassManager for MILP routing stage.

        Args:
            pass_manager_config: Pass manager configuration object
            optimization_level: Optimisation level (0-3)

        Returns:
            PassManager for MILP routing stage
        """
        routing_pm = PassManager()
        # Add MILP routing pass
        routing_pm.append(MilpRouting(pass_manager_config.coupling_map))

        return routing_pm


class QuariadneLpLayoutPlugin(PassManagerStagePlugin):
    """PassManager stage plugin for LP-based layout optimisation with Birkhoff decomposition."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        """Generate PassManager for LP layout stage.

        Args:
            pass_manager_config: Pass manager configuration object
            optimization_level: Optimisation level (0-3)

        Returns:
            PassManager for LP layout stage
        """
        layout_pm = PassManager()

        # Add LP layout pass
        if (
            hasattr(pass_manager_config, "target")
            and pass_manager_config.target is not None
        ):
            layout_pm.append(
                MilpLayout(pass_manager_config.target, mode=RoutingMode.LP)
            )
        else:
            layout_pm.append(
                MilpLayout(pass_manager_config.coupling_map, mode=RoutingMode.LP)
            )

        # Embed the layout using generate_embed_passmanager
        embed_pm = generate_embed_passmanager(pass_manager_config.coupling_map)
        layout_pm += embed_pm

        return layout_pm


class QuariadneLpRoutingPlugin(PassManagerStagePlugin):
    """PassManager stage plugin for LP-based routing optimisation with Birkhoff decomposition."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        """Generate PassManager for LP routing stage.

        Args:
            pass_manager_config: Pass manager configuration object
            optimization_level: Optimisation level (0-3)

        Returns:
            PassManager for LP routing stage
        """
        routing_pm = PassManager()
        # Add LP routing pass
        routing_pm.append(MilpRouting(pass_manager_config.coupling_map))

        return routing_pm


class QuariadneEdgeLayoutPlugin(PassManagerStagePlugin):
    """PassManager stage plugin for Edge-based layout optimisation with bipartite matching."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        """Generate PassManager for Edge layout stage.

        Args:
            pass_manager_config: Pass manager configuration object
            optimization_level: Optimisation level (0-3)

        Returns:
            PassManager for Edge layout stage
        """
        layout_pm = PassManager()

        # Add Edge layout pass
        if (
            hasattr(pass_manager_config, "target")
            and pass_manager_config.target is not None
        ):
            layout_pm.append(
                MilpLayout(pass_manager_config.target, mode=RoutingMode.EDGE)
            )
        else:
            layout_pm.append(
                MilpLayout(pass_manager_config.coupling_map, mode=RoutingMode.EDGE)
            )

        # Embed the layout using generate_embed_passmanager
        embed_pm = generate_embed_passmanager(pass_manager_config.coupling_map)
        layout_pm += embed_pm

        return layout_pm


class QuariadneEdgeRoutingPlugin(PassManagerStagePlugin):
    """PassManager stage plugin for Edge-based routing optimisation with bipartite matching."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        """Generate PassManager for Edge routing stage.

        Args:
            pass_manager_config: Pass manager configuration object
            optimization_level: Optimisation level (0-3)

        Returns:
            PassManager for Edge routing stage
        """
        routing_pm = PassManager()
        # Add Edge routing pass
        routing_pm.append(MilpRouting(pass_manager_config.coupling_map))

        return routing_pm


class BipartiteLayout(qiskit.transpiler.AnalysisPass):
    """Layout pass using bipartite allocation LP for initial qubit placement."""

    def __init__(
        self,
        coupling_map: Union[qiskit.transpiler.CouplingMap, qiskit.transpiler.Target],
    ) -> None:
        """Initialise bipartite layout pass with backend coupling constraints.

        Args:
            coupling_map: Backend coupling map defining physical qubit connectivity
        """
        super().__init__()
        if isinstance(coupling_map, qiskit.transpiler.Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map

        self.coupling_graph = quariadne.bipartite_allocation_lp.get_coupling_graph(
            self.coupling_map
        )

    def _convert_dag_to_circuit(
        self, dag: qiskit.dagcircuit.DAGCircuit
    ) -> quariadne.circuit.AbstractQuantumCircuit:
        """Convert Qiskit DAG to Quariadne abstract quantum circuit representation.

        Args:
            dag: Input Qiskit DAG circuit to convert

        Returns:
            Quariadne abstract quantum circuit for LP optimisation
        """
        quariadne_dag = quariadne.computational_graph.ComputationalDAG.from_qiskit_dag(
            dag
        )
        return quariadne_dag.to_abstract_quantum_circuit()

    def _generate_physical_qubit_indices(
        self,
        physical_by_logical_mapping: Dict[
            quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit
        ],
        dag_qubit_count: int,
    ) -> List[int]:
        """Extract physical qubit indices from LP mapping solution for layout creation.

        Args:
            physical_by_logical_mapping: LP solution mapping logical to physical qubits
            dag_qubit_count: Number of qubits in the original DAG circuit

        Returns:
            List of physical qubit indices ordered by logical qubit index
        """
        sorted_logical_qubits = sorted(
            physical_by_logical_mapping.keys(), key=lambda qubit: qubit.index
        )

        valid_logical_qubits = [
            qubit for qubit in sorted_logical_qubits if qubit.index < dag_qubit_count
        ]

        physical_qubit_indices = []
        for logical_qubit in valid_logical_qubits:
            physical_qubit = physical_by_logical_mapping[logical_qubit]
            physical_qubit_indices.append(physical_qubit.index)

        return physical_qubit_indices

    def run(self, dag: qiskit.dagcircuit.DAGCircuit) -> None:
        """Execute bipartite LP optimisation to determine optimal initial qubit layout."""
        if self.target is not None:
            if dag.num_qubits() > self.target.num_qubits:
                raise qiskit.transpiler.TranspilerError(
                    "Number of qubits greater than device."
                )
        elif dag.num_qubits() > self.coupling_map.size():
            raise qiskit.transpiler.TranspilerError(
                "Number of qubits greater than device."
            )

        quariadne_circuit = self._convert_dag_to_circuit(dag)
        bipartite_router = quariadne.bipartite_allocation_lp.BipartiteAllocationRouter(
            self.coupling_graph, quariadne_circuit
        )
        bipartite_result = bipartite_router.run()

        self.property_set["bipartite_result"] = bipartite_result
        physical_by_logical_initial_mapping = bipartite_result.initial_mapping

        physical_qubit_indices = self._generate_physical_qubit_indices(
            physical_by_logical_initial_mapping, dag.num_qubits()
        )

        canonical_register = dag.qregs["q"]
        bipartite_layout = qiskit.transpiler.Layout.from_intlist(
            physical_qubit_indices, canonical_register
        )
        self.property_set["layout"] = bipartite_layout


class BipartiteRouting(qiskit.transpiler.TransformationPass):
    """Routing pass using bipartite allocation LP for SWAP insertion.

    Computes swap sequences from LP edge assignments to route operations
    to their assigned edges on the coupling map.
    """

    def __init__(self, coupling_map: qiskit.transpiler.CouplingMap) -> None:
        """Initialise bipartite routing pass with backend coupling constraints.

        Args:
            coupling_map: Backend coupling map defining physical qubit connectivity.
        """
        super().__init__()
        self.coupling_map = coupling_map

    def _create_swap_dag(
        self,
        canonical_register: qiskit.circuit.QuantumRegister,
        current_layout: qiskit.transpiler.Layout,
        swaps: List[quariadne.circuit.PhysicalSwap],
    ) -> qiskit.dagcircuit.DAGCircuit:
        """Create DAG containing SWAP operations for a specific operation.

        Follows the pattern from MilpRouting._create_swap_dag.

        Args:
            canonical_register: Quantum register for the circuit.
            current_layout: Current mapping of qubits to physical positions.
            swaps: List of physical qubit pairs that need to be swapped.

        Returns:
            DAG circuit containing all required SWAP operations.
        """
        swap_dag = qiskit.dagcircuit.DAGCircuit()
        swap_dag.add_qreg(canonical_register)

        # Add each required SWAP operation to the DAG
        for swap in swaps:
            physical_qubit_1 = swap.first.index
            physical_qubit_2 = swap.second.index
            logical_qubit_1 = current_layout[physical_qubit_1]
            logical_qubit_2 = current_layout[physical_qubit_2]

            swap_dag.apply_operation_back(
                qiskit.circuit.library.SwapGate(),
                (logical_qubit_1, logical_qubit_2),
                cargs=(),
                check=False,
            )

        return swap_dag

    def _apply_swaps_to_layout(
        self,
        current_layout: qiskit.transpiler.Layout,
        swaps: List[quariadne.circuit.PhysicalSwap],
    ) -> qiskit.transpiler.Layout:
        """Apply SWAP operations to layout and return updated layout.

        Follows the pattern from MilpRouting._apply_swaps_to_layout.

        Args:
            current_layout: Current mapping of qubits to physical positions.
            swaps: List of physical qubit pairs that need to be swapped.

        Returns:
            New layout with SWAP operations applied.
        """
        updated_layout = current_layout.copy()

        for swap in swaps:
            physical_qubit_1 = swap.first.index
            physical_qubit_2 = swap.second.index
            updated_layout.swap(physical_qubit_1, physical_qubit_2)

        return updated_layout

    def _compute_swaps_for_edge(
        self,
        current_layout: qiskit.transpiler.Layout,
        q0_phys: int,
        q1_phys: int,
        required_left: int,
        required_right: int,
    ) -> List[quariadne.circuit.PhysicalSwap]:
        """Compute swaps to move qubits to required edge positions.

        Uses iterative approach: move each qubit to its required position,
        re-checking after each move since swaps can displace the other qubit.

        Args:
            current_layout: Current layout state.
            q0_phys: Current physical position of left qubit.
            q1_phys: Current physical position of right qubit.
            required_left: Required position for left qubit.
            required_right: Required position for right qubit.

        Returns:
            List of PhysicalSwap operations.
        """
        import networkx as nx

        swaps: List[quariadne.circuit.PhysicalSwap] = []

        # Build graph for shortest path computation
        graph = nx.Graph()
        graph.add_edges_from(self.coupling_map.get_edges())

        # Track positions (will be modified by swaps)
        pos_left = q0_phys
        pos_right = q1_phys

        # Iterate until both qubits are in position
        for _ in range(20):  # Safety limit
            if pos_left == required_left and pos_right == required_right:
                break

            # Move left qubit if needed
            if pos_left != required_left:
                try:
                    path = nx.shortest_path(graph, pos_left, required_left)
                    for i in range(len(path) - 1):
                        swaps.append(
                            quariadne.circuit.PhysicalSwap(
                                quariadne.circuit.PhysicalQubit(path[i]),
                                quariadne.circuit.PhysicalQubit(path[i + 1]),
                            )
                        )
                        # Update tracked positions
                        if pos_right == path[i + 1]:
                            pos_right = path[i]
                        pos_left = path[i + 1]
                except nx.NetworkXNoPath:
                    break

            # Move right qubit if needed
            if pos_right != required_right:
                try:
                    path = nx.shortest_path(graph, pos_right, required_right)
                    for i in range(len(path) - 1):
                        swaps.append(
                            quariadne.circuit.PhysicalSwap(
                                quariadne.circuit.PhysicalQubit(path[i]),
                                quariadne.circuit.PhysicalQubit(path[i + 1]),
                            )
                        )
                        # Update tracked positions
                        if pos_left == path[i + 1]:
                            pos_left = path[i]
                        pos_right = path[i + 1]
                except nx.NetworkXNoPath:
                    break

        return swaps

    def run(self, dag: qiskit.dagcircuit.DAGCircuit) -> qiskit.dagcircuit.DAGCircuit:
        """Apply bipartite LP-determined SWAP operations to the DAG.

        Computes swaps on-the-fly based on the required edge for each gate
        and the current layout state. This is more robust than using precomputed
        swaps since it always uses the actual current mapping.
        """
        new_dag = dag.copy_empty_like()

        if self.coupling_map is None:
            raise qiskit.transpiler.TranspilerError(
                "BipartiteRouting cannot run with coupling_map=None"
            )

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise qiskit.transpiler.TranspilerError(
                "Bipartite routing runs on physical circuits only"
            )

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise qiskit.transpiler.TranspilerError(
                "Layout does not match DAG qubit count"
            )

        # Retrieve LP result from property set (populated by BipartiteLayout)
        if "bipartite_result" not in self.property_set:
            raise qiskit.transpiler.TranspilerError(
                "BipartiteRouting requires BipartiteLayout to be run first"
            )

        canonical_register = dag.qregs["q"]
        bipartite_result = self.property_set["bipartite_result"]
        edge_by_qubits = bipartite_result.operation_to_edge_by_qubits

        # Initialise layout tracking (trivial after embedding)
        trivial_layout = qiskit.transpiler.Layout.generate_trivial_layout(
            canonical_register
        )
        current_layout = trivial_layout.copy()

        # Track qubit pair occurrence counts for operation matching
        qubit_pair_counts: Dict[tuple, int] = {}

        # Process each layer
        for layer in dag.serial_layers():
            subdag = layer["graph"]

            for gate in subdag.two_qubit_ops():
                # Physical qubit indices from the embedded DAG
                q0_phys = gate.qargs[0]._index
                q1_phys = gate.qargs[1]._index
                qubit_pair_key = (q0_phys, q1_phys)

                occurrence = qubit_pair_counts.get(qubit_pair_key, 0)
                qubit_pair_counts[qubit_pair_key] = occurrence + 1
                full_key = (q0_phys, q1_phys, occurrence)

                # Get required edge from LP result
                if full_key in edge_by_qubits:
                    required_edge = edge_by_qubits[full_key]
                    required_left = required_edge[0].index
                    required_right = required_edge[1].index

                    # Current positions: look up in current_layout since swaps
                    # may have moved qubits from their embedded positions.
                    # gate.qargs[i] is a Qubit object; current_layout maps it
                    # to its current physical position.
                    current_left = current_layout[gate.qargs[0]]
                    current_right = current_layout[gate.qargs[1]]

                    # Check if gate is already on the required edge
                    on_required_edge = (
                        current_left == required_left
                        and current_right == required_right
                    ) or (
                        current_left == required_right
                        and current_right == required_left
                    )
                    if not on_required_edge:
                        swaps = self._compute_swaps_for_edge(
                            current_layout,
                            current_left,
                            current_right,
                            required_left,
                            required_right,
                        )
                        if swaps:
                            swap_dag = self._create_swap_dag(
                                canonical_register, current_layout, swaps
                            )
                            qubit_order = current_layout.reorder_bits(new_dag.qubits)
                            new_dag.compose(swap_dag, qubits=qubit_order)
                            current_layout = self._apply_swaps_to_layout(
                                current_layout, swaps
                            )

            # Add the entire layer using current layout
            qubit_order = current_layout.reorder_bits(new_dag.qubits)
            new_dag.compose(subdag, qubits=qubit_order)

        # Update final layout state in property set
        if self.property_set["final_layout"] is None:
            self.property_set["final_layout"] = current_layout
        else:
            self.property_set["final_layout"] = self.property_set[
                "final_layout"
            ].compose(current_layout, dag.qubits)

        return new_dag


class QuariadneBipartiteLayoutPlugin(PassManagerStagePlugin):
    """PassManager stage plugin for bipartite LP-based layout optimisation."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        """Generate PassManager for bipartite layout stage.

        Args:
            pass_manager_config: Pass manager configuration object
            optimization_level: Optimisation level (0-3)

        Returns:
            PassManager for bipartite layout stage
        """
        layout_pm = PassManager()

        if (
            hasattr(pass_manager_config, "target")
            and pass_manager_config.target is not None
        ):
            layout_pm.append(BipartiteLayout(pass_manager_config.target))
        else:
            layout_pm.append(BipartiteLayout(pass_manager_config.coupling_map))

        embed_pm = generate_embed_passmanager(pass_manager_config.coupling_map)
        layout_pm += embed_pm

        return layout_pm


class QuariadneBipartiteRoutingPlugin(PassManagerStagePlugin):
    """PassManager stage plugin for bipartite LP-based routing optimisation."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        """Generate PassManager for bipartite routing stage.

        Args:
            pass_manager_config: Pass manager configuration object
            optimization_level: Optimisation level (0-3)

        Returns:
            PassManager for bipartite routing stage
        """
        routing_pm = PassManager()
        routing_pm.append(BipartiteRouting(pass_manager_config.coupling_map))

        return routing_pm
