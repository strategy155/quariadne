"""Layer-by-layer LP router for quantum circuit routing.

This module implements an iterative LP-based routing strategy that processes
one layer at a time, solving a global LP for the remaining circuit at each
iteration while fixing initial positions from the previous layer's final mapping.

Algorithm Overview
------------------
1. Layer 0: Solve global LP for entire circuit -> extract mapping after layer 0
2. Layer N: Remove layer N-1, solve global LP for remaining circuit with
   initial positions fixed from mapping after layer N-1
3. Accumulate swaps across all iterations

References
----------
- HiGHS Python Interface: https://ergo-code.github.io/HiGHS/dev/interfaces/python/
- Python dataclasses: https://docs.python.org/3/library/dataclasses.html
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import networkx as nx

import quariadne.circuit
from quariadne.lp_router_base import (
    LPSolverOptions,
    PhysicalEdge,
)
from quariadne.lp_router_unified import LpRouterUnified, UnifiedLPResult

_logger = logging.getLogger(__name__)


@dataclass
class LayeredLPResult:
    """Result container for layered LP router optimisation.

    Attributes:
        objective_value: Sum of objective values across all layer iterations.
        initial_mapping: Logical-to-physical mapping from first layer's solve.
        final_mapping: Logical-to-physical mapping from last layer's solve.
        operations: List of two-qubit operations that were routed.
        inserted_swaps: Dict mapping operation index to list of PhysicalSwap.
        layer_results: Per-layer objective values for analysis.
        layer_count: Total number of layers processed.
    """

    objective_value: float
    initial_mapping: dict[
        quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit
    ]
    final_mapping: dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]
    operations: list[quariadne.circuit.QuantumOperation]
    inserted_swaps: dict[int, list[quariadne.circuit.PhysicalSwap]]
    layer_results: list[float]
    layer_count: int


class LpRouterLayered:
    """Layer-by-layer iterative LP router for quantum circuit routing.

    Processes the circuit one layer at a time, solving a global LP for the
    remaining subcircuit at each iteration. Initial positions are fixed from
    the previous layer's mapping after that layer completes.

    Attributes:
        coupling_map: NetworkX DiGraph representing hardware connectivity.
        quantum_circuit: The quantum circuit to route.
        solver_options: Configuration for HiGHS LP solver.
        operations: List of two-qubit operations extracted from circuit.
        operation_layers: Operations grouped into parallel layers.
        edges: List of physical edges (coupling map edges).
        edge_to_index: Lookup from edge tuple to list index.
    """

    def __init__(
        self,
        coupling_map: nx.DiGraph[quariadne.circuit.PhysicalQubit],
        quantum_circuit: quariadne.circuit.AbstractQuantumCircuit,
        solver_options: LPSolverOptions | None = None,
        propagate_constraints: bool = False,  # Disabled by default until debugged
    ) -> None:
        """Initialise the layered LP router.

        Args:
            coupling_map: Hardware connectivity graph with PhysicalQubit nodes.
            quantum_circuit: Circuit to route (will be deepcopied internally).
            solver_options: Optional HiGHS solver configuration.
            propagate_constraints: Whether to propagate position constraints
                between layers. Disabled by default due to potential infeasibility.
        """
        self.coupling_map = coupling_map
        self.quantum_circuit = copy.deepcopy(quantum_circuit)
        self.solver_options = solver_options or LPSolverOptions()
        self.propagate_constraints = propagate_constraints

        self._add_dummy_qubits()

        self.operations = self._get_two_qubit_operations()
        self.operation_count = len(self.operations)

        self.operation_layers = self._build_operation_layers()

        self.edges: list[PhysicalEdge] = list(self.coupling_map.edges())
        self.edge_count = len(self.edges)
        self.edge_to_index: dict[PhysicalEdge, int] = {
            edge: idx for idx, edge in enumerate(self.edges)
        }

        _logger.info(
            "LpRouterLayered initialised: %d operations, %d layers, %d edges",
            self.operation_count,
            len(self.operation_layers),
            self.edge_count,
        )

    def _add_dummy_qubits(self) -> None:
        """Add dummy qubits to match hardware qubit count."""
        hardware_qubit_count = self.coupling_map.number_of_nodes()
        circuit_qubit_count = len(self.quantum_circuit.qubits)

        if circuit_qubit_count < hardware_qubit_count:
            dummy_qubits = tuple(
                quariadne.circuit.LogicalQubit(i)
                for i in range(circuit_qubit_count, hardware_qubit_count)
            )
            self.quantum_circuit.qubits = self.quantum_circuit.qubits + dummy_qubits

    def _get_two_qubit_operations(self) -> list[quariadne.circuit.QuantumOperation]:
        """Extract two-qubit operations from the circuit."""
        return [
            op
            for op in self.quantum_circuit.operations
            if len(op.qubits_participating) == 2
        ]

    def _build_operation_layers(self) -> list[list[int]]:
        """Group operations into parallel layers based on qubit dependencies.

        Operations in the same layer do not share any qubits and can execute
        in parallel. Uses greedy assignment to earliest possible layer.

        The algorithm tracks, for each qubit, the last layer where it was used.
        Each new operation is assigned to layer = max(last_layer[q] + 1) for
        all its participating qubits. This ensures no qubit conflicts within
        a layer.

        Returns:
            List of layers, each layer is a list of operation indices.

        Reference:
            https://docs.python.org/3/library/stdtypes.html#dict
        """
        if self.operation_count == 0:
            return []

        layers: list[list[int]] = []
        # Track the most recent layer where each qubit was involved
        qubit_last_layer: dict[quariadne.circuit.LogicalQubit, int] = {}

        for operation_index, operation in enumerate(self.operations):
            qubits = operation.qubits_participating

            # Find the minimum layer this operation can be placed in.
            # Must be after all previous operations involving any of its qubits.
            min_layer = 0
            for qubit in qubits:
                if qubit in qubit_last_layer:
                    # This qubit was used in qubit_last_layer[qubit], so this
                    # operation must go in at least the next layer.
                    min_layer = max(min_layer, qubit_last_layer[qubit] + 1)

            # Extend layers list if we need a new layer
            while len(layers) <= min_layer:
                layers.append([])

            # Place operation in its layer
            layers[min_layer].append(operation_index)

            # Update tracking: all qubits in this operation are now "used" in min_layer
            for qubit in qubits:
                qubit_last_layer[qubit] = min_layer

        return layers

    def _extract_mapping_after_layer(
        self,
        result: UnifiedLPResult,
        layer_ops_local: list[int],
        subcircuit_operations: list[quariadne.circuit.QuantumOperation],
    ) -> dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]:
        """Extract qubit positions after current layer's operations complete.

        This is NOT the same as result.final_mapping, which gives positions
        after ALL operations. We need positions after just the current layer,
        to use as initial constraints for the next iteration.

        For each qubit:
        - If qubit has operations in this layer: position comes from the
          LAST operation in the layer (where qubit ends up after the layer)
        - If qubit has no operations in this layer: position is unchanged
          from initial_mapping (qubit didn't move during this layer)

        Args:
            result: LP result from LpRouterUnified.run().
            layer_ops_local: Local indices of operations in the current layer.
            subcircuit_operations: Operations list for the current subcircuit.

        Returns:
            Mapping of logical qubits to physical positions after the layer.
        """
        # Start with initial mapping - positions at START of this subcircuit.
        # Qubits that don't have operations in current layer keep these positions.
        mapping = dict(result.initial_mapping)

        # For qubits that DO have operations in the current layer, we need to
        # find their LAST operation in the layer to determine final position.
        # Build qubit -> last operation index in this layer.
        qubit_last_op_in_layer: dict[quariadne.circuit.LogicalQubit, int] = {}
        for local_idx in layer_ops_local:
            op = subcircuit_operations[local_idx]
            for qubit in op.qubits_participating:
                # Overwrite with each operation; final value is the last one
                qubit_last_op_in_layer[qubit] = local_idx

        # Update positions for qubits that had operations in this layer.
        # Position is determined by which edge the last operation was assigned to.
        for qubit, last_op_idx in qubit_last_op_in_layer.items():
            edge = result.operation_edge_assignment[last_op_idx]
            op = subcircuit_operations[last_op_idx]
            left_qubit, right_qubit = op.qubits_participating

            # Edge is (left_physical, right_physical).
            # Left qubit of operation sits at edge[0], right qubit at edge[1].
            if qubit == left_qubit:
                mapping[qubit] = edge[0]
            else:
                mapping[qubit] = edge[1]

        return mapping

    def _create_subcircuit_from_layer(
        self,
        start_layer_idx: int,
    ) -> tuple[
        quariadne.circuit.AbstractQuantumCircuit,
        dict[int, int],
        dict[int, int],
    ]:
        """Create subcircuit containing operations from start_layer onwards.

        Each iteration of the layered router works on a subcircuit that contains
        only the remaining operations (from current layer to the end). This
        method builds that subcircuit and provides index mappings.

        The mappings are needed because:
        - LpRouterUnified operates on local indices (0, 1, 2, ...)
        - We need to translate back to global indices to accumulate swaps
        - We need to translate global layer indices to local for constraints

        Args:
            start_layer_idx: Index of the first layer to include in subcircuit.

        Returns:
            Tuple of (subcircuit, global_to_local, local_to_global):
            - subcircuit: New AbstractQuantumCircuit with selected operations
            - global_to_local: Maps original operation index to subcircuit index
            - local_to_global: Maps subcircuit index to original operation index
        """
        # Collect operation indices for all layers from start_layer_idx onwards.
        # These are GLOBAL indices into self.operations.
        global_indices: list[int] = []
        for layer_idx in range(start_layer_idx, len(self.operation_layers)):
            global_indices.extend(self.operation_layers[layer_idx])

        # Build bidirectional index mappings.
        # global_to_local[g] = local means operation g in original becomes local in subcircuit
        # local_to_global[local] = g means operation local in subcircuit is g in original
        global_to_local = {g: local for local, g in enumerate(global_indices)}
        local_to_global = {local: g for local, g in enumerate(global_indices)}

        # Create the subcircuit with only the selected operations.
        # Keep the same qubits (including dummy qubits) to maintain consistency.
        selected_ops = [self.operations[g] for g in global_indices]
        subcircuit = quariadne.circuit.AbstractQuantumCircuit(
            qubits=self.quantum_circuit.qubits,
            operations=selected_ops,
        )

        return subcircuit, global_to_local, local_to_global

    def _extract_layer_swaps(
        self,
        result: UnifiedLPResult,
        current_layer_local_indices: set[int],
        local_to_global: dict[int, int],
    ) -> dict[int, list[quariadne.circuit.PhysicalSwap]]:
        """Extract swaps for current layer operations only.

        The LP result (result.inserted_swaps) contains swaps for ALL transitions
        in the subcircuit - including swaps for future layers. But we only want
        swaps that occur BEFORE operations in the current layer, because:
        - Swaps for future layers will be re-computed in future iterations
        - We only commit swaps for the layer we're currently processing

        This filtering ensures each layer's swaps are computed independently
        with the correct starting positions.

        Args:
            result: LP result from LpRouterUnified.run().
            current_layer_local_indices: Set of local indices for current layer ops.
            local_to_global: Map from subcircuit indices to original indices.

        Returns:
            Dict mapping GLOBAL operation index to list of PhysicalSwap objects.
        """
        layer_swaps: dict[int, list[quariadne.circuit.PhysicalSwap]] = {}

        # Filter: only include swaps for operations in the current layer
        for local_idx, swaps in result.inserted_swaps.items():
            if local_idx in current_layer_local_indices:
                # Translate local index back to global
                global_idx = local_to_global[local_idx]
                if swaps:
                    layer_swaps[global_idx] = swaps

        return layer_swaps

    def run(self) -> LayeredLPResult:
        """Run layered LP routing on the circuit.

        Main algorithm:
        1. For layer 0: solve global LP for entire circuit
        2. Extract mapping after layer 0 (where qubits end up)
        3. For subsequent layers: peel off previous layer, solve global LP
           for remaining circuit with initial positions fixed
        4. Accumulate swaps across all iterations

        Returns:
            LayeredLPResult containing the complete routing solution.
        """
        # Handle empty circuit - nothing to route
        if self.operation_count == 0:
            _logger.info("Empty circuit, nothing to route")
            return LayeredLPResult(
                objective_value=0.0,
                initial_mapping={},
                final_mapping={},
                operations=[],
                inserted_swaps={},
                layer_results=[],
                layer_count=0,
            )

        # Handle single layer - no iteration needed, delegate directly
        if len(self.operation_layers) == 1:
            _logger.info("Single layer circuit, delegating to LpRouterUnified")
            return self._solve_single_layer()

        # Multi-layer circuit - use iterative approach
        return self._solve_iteratively()

    def _solve_single_layer(self) -> LayeredLPResult:
        """Handle circuit with only one layer - no iteration needed."""
        router = LpRouterUnified(
            coupling_map=self.coupling_map,
            quantum_circuit=self.quantum_circuit,
            solver_options=self.solver_options,
        )
        result = router.run()

        return LayeredLPResult(
            objective_value=result.objective_value,
            initial_mapping=result.initial_mapping,
            final_mapping=result.final_mapping,
            operations=self.operations,
            inserted_swaps=result.inserted_swaps,
            layer_results=[result.objective_value],
            layer_count=1,
        )

    def _solve_iteratively(self) -> LayeredLPResult:
        """Solve circuit layer-by-layer with iterative global LP.

        Core algorithm:
        - Each iteration solves a GLOBAL LP for the remaining circuit
        - The "remaining circuit" starts from current layer to the end
        - Initial positions are fixed from previous layer's end positions
        - Only swaps for the current layer are extracted and accumulated
        """
        all_swaps: dict[int, list[quariadne.circuit.PhysicalSwap]] = {}
        layer_objectives: list[float] = []

        # These will be populated during iteration
        initial_mapping: dict[
            quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit
        ] = {}
        layer_end_mapping: dict[
            quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit
        ] = {}

        num_layers = len(self.operation_layers)

        for layer_idx in range(num_layers):
            _logger.info("Processing layer %d/%d", layer_idx + 1, num_layers)

            # Step 1: Create subcircuit from this layer onwards.
            # This contains all operations from layer_idx to the end.
            subcircuit, global_to_local, local_to_global = (
                self._create_subcircuit_from_layer(layer_idx)
            )

            # Step 2: Get local indices for operations in the FIRST layer
            # of this subcircuit (which is our current layer in global terms).
            first_layer_ops_global = self.operation_layers[layer_idx]
            first_layer_ops_local = [global_to_local[g] for g in first_layer_ops_global]

            # Step 3: Build initial_position_mapping from previous layer's end mapping.
            # For layer 0, there's no previous mapping - LP solves freely.
            # For layer N > 0, we constrain initial qubit positions based on where
            # qubits ended up after layer N-1.
            #
            # IMPORTANT: We use initial_position_mapping (per-qubit constraints)
            # instead of fixed_operation_edges (per-operation constraints). This is
            # more flexible because:
            # - Each qubit's position is constrained independently
            # - Partner qubits in operations can be at any adjacent position
            # - Avoids infeasibility from over-constraining both qubits simultaneously
            #
            # Reference: Plan file sleepy-beaming-kay.md, Thesis 03-method.tex
            initial_pos_mapping: (
                dict[quariadne.circuit.LogicalQubit, quariadne.circuit.PhysicalQubit]
                | None
            ) = None

            if self.propagate_constraints and layer_idx > 0 and layer_end_mapping:
                initial_pos_mapping = layer_end_mapping
                _logger.debug(
                    "Propagating %d qubit positions from previous layer",
                    len(initial_pos_mapping),
                )

            # Step 4: Solve global LP for the subcircuit.
            # This optimises over ALL remaining operations, but with
            # initial positions constrained (for layer_idx > 0).
            router = LpRouterUnified(
                coupling_map=self.coupling_map,
                quantum_circuit=subcircuit,
                solver_options=self.solver_options,
                initial_position_mapping=initial_pos_mapping,
            )
            result = router.run()
            layer_objectives.append(result.objective_value)

            # Step 5: Store initial mapping from first layer only.
            # This is the overall initial mapping for the full circuit.
            if layer_idx == 0:
                initial_mapping = result.initial_mapping

            # Step 6: Extract mapping AFTER current layer.
            # This is where qubits end up after this layer's operations,
            # which becomes the initial constraint for the next iteration.
            layer_end_mapping = self._extract_mapping_after_layer(
                result,
                first_layer_ops_local,
                subcircuit.operations,
            )

            # Step 7: Extract swaps for CURRENT layer only.
            # We don't want swaps for future layers - those will be
            # re-computed in future iterations with correct constraints.
            first_layer_local_set = set(first_layer_ops_local)
            layer_swaps = self._extract_layer_swaps(
                result,
                first_layer_local_set,
                local_to_global,
            )

            # Step 8: Accumulate swaps into overall result.
            # Use global operation indices as keys.
            for global_idx, swaps in layer_swaps.items():
                if global_idx not in all_swaps:
                    all_swaps[global_idx] = []
                for swap in swaps:
                    # Avoid duplicates (shouldn't happen, but defensive)
                    if swap not in all_swaps[global_idx]:
                        all_swaps[global_idx].append(swap)

            _logger.debug(
                "Layer %d: objective=%.4f, swaps=%d",
                layer_idx,
                result.objective_value,
                sum(len(s) for s in layer_swaps.values()),
            )

        # Final mapping is the layer_end_mapping from the last iteration
        final_mapping = layer_end_mapping

        total_swaps = sum(len(s) for s in all_swaps.values())
        _logger.info(
            "Layered routing complete: %d layers, %d total SWAPs, objective=%.4f",
            num_layers,
            total_swaps,
            sum(layer_objectives),
        )

        return LayeredLPResult(
            objective_value=sum(layer_objectives),
            initial_mapping=initial_mapping,
            final_mapping=final_mapping,
            operations=self.operations,
            inserted_swaps=all_swaps,
            layer_results=layer_objectives,
            layer_count=num_layers,
        )
