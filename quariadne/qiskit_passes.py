from typing import List, Dict, Union
import quariadne.computational_graph
import quariadne.milp_router
import quariadne.circuit
import qiskit.transpiler
import qiskit.dagcircuit
import qiskit.circuit.library
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.preset_passmanagers.common import generate_embed_passmanager
from qiskit.transpiler import PassManager


class MilpLayout(qiskit.transpiler.AnalysisPass):
    """Layout pass using MILP optimization for initial qubit placement."""

    def __init__(
        self,
        coupling_map: Union[qiskit.transpiler.CouplingMap, qiskit.transpiler.Target],
    ) -> None:
        """Initialise MILP layout pass with backend coupling constraints.

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

        self.coupling_graph = quariadne.milp_router.get_coupling_graph(
            self.coupling_map
        )

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
        """Execute MILP optimization to determine optimal initial qubit layout."""
        if self.target is not None:
            if dag.num_qubits() > self.target.num_qubits:
                raise qiskit.transpiler.TranspilerError(
                    "Number of qubits greater than device."
                )
        elif dag.num_qubits() > self.coupling_map.size():
            raise qiskit.transpiler.TranspilerError(
                "Number of qubits greater than device."
            )

        # Convert to internal representation and solve MILP
        quariadne_circuit = self._convert_dag_to_circuit(dag)
        milp_router = quariadne.milp_router.MilpRouter(
            self.coupling_graph, quariadne_circuit
        )
        milp_router_result = milp_router.run()

        # Store result for potential use by routing pass
        self.property_set["milp_result"] = milp_router_result
        physical_by_logical_initial_mapping = milp_router_result.initial_mapping

        # Generate layout from MILP solution
        physical_qubit_indices = self._generate_physical_qubit_indices(
            physical_by_logical_initial_mapping, dag.num_qubits()
        )

        # generation of the final layout
        canonical_register = dag.qregs["q"]
        milp_layout = qiskit.transpiler.Layout.from_intlist(
            physical_qubit_indices, canonical_register
        )
        self.property_set["layout"] = milp_layout


class MilpRouting(qiskit.transpiler.TransformationPass):
    """Routing pass using MILP optimization for SWAP insertion."""

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

        # Retrieve MILP result from property set (populated by MilpLayout)
        if "milp_result" not in self.property_set:
            raise qiskit.transpiler.TranspilerError(
                "MilpRouting requires MilpLayout to be run first"
            )

        canonical_register = dag.qregs["q"]
        milp_router_result = self.property_set["milp_result"]
        swap_pairs_by_timestep = milp_router_result.inserted_swaps

        # Initialise layout tracking
        trivial_layout = qiskit.transpiler.Layout.generate_trivial_layout(
            canonical_register
        )
        current_layout = trivial_layout.copy()

        two_qubit_idx = 0
        # Process each layer and insert SWAPs as determined by MILP
        for layer in dag.serial_layers():
            subdag = layer["graph"]

            # Insert SWAPs before processing gates if required at this timestep
            for gate in subdag.two_qubit_ops():
                if two_qubit_idx in swap_pairs_by_timestep:
                    swaps = swap_pairs_by_timestep[two_qubit_idx]
                    # Create and insert SWAP operations
                    swap_dag = self._create_swap_dag(
                        canonical_register, current_layout, swaps
                    )
                    order = current_layout.reorder_bits(new_dag.qubits)
                    new_dag.compose(swap_dag, qubits=order)

                    # Update layout state after SWAPs
                    current_layout = self._apply_swaps_to_layout(current_layout, swaps)

                two_qubit_idx += 1

            # Add the original layer operations
            order = current_layout.reorder_bits(new_dag.qubits)
            new_dag.compose(subdag, qubits=order)

        # Update final layout state in property set
        if self.property_set["final_layout"] is None:
            self.property_set["final_layout"] = current_layout
        else:
            self.property_set["final_layout"] = self.property_set[
                "final_layout"
            ].compose(current_layout, dag.qubits)

        return new_dag


class QuariadneMilpLayoutPlugin(PassManagerStagePlugin):
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


class QuariadneMilpRoutingPlugin(PassManagerStagePlugin):
    """PassManager stage plugin for MILP-based routing optimization."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        """Generate PassManager for MILP routing stage.

        Args:
            pass_manager_config: Pass manager configuration object
            optimization_level: Optimization level (0-3)

        Returns:
            PassManager for MILP routing stage
        """
        routing_pm = PassManager()
        # Add MILP routing pass
        routing_pm.append(MilpRouting(pass_manager_config.coupling_map))

        return routing_pm
