import dataclasses
import typing
import networkx as nx
import abc
import matplotlib.pyplot as plt
import qiskit
import qiskit.dagcircuit

import quariadne.circuit


# This type represents all the DAG nodes, which are meaningful for our computations
type QiskitDAGNode = typing.Union[
    qiskit.dagcircuit.DAGOpNode,
    qiskit.dagcircuit.DAGInNode,
    qiskit.dagcircuit.DAGOutNode,
]

# This type is a mapping of what qiskit library returns by the edges method on the DAG circuit.
type QiskitDAGEdge = typing.Tuple[QiskitDAGNode, QiskitDAGNode, qiskit.circuit.Qubit]


# This type represents the special format of the nx edges
type NXEdge = typing.Tuple[ComputationalNode, ComputationalNode]

DAG_VISUALISATION_TITLE = "Visualised DAG"


@dataclasses.dataclass(frozen=True)
class ComputationalNode(abc.ABC):
    """Base class for all nodes in the internal DAG representation of the routing circuit."""

    @property
    @abc.abstractmethod
    def label(self):
        """Return the label of the node for plotting purposes.

        Returns:
            str: The label to be used for plotting this node.
        """
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class WireStart(ComputationalNode):
    """Represents the start of a DAG wire.

    Attributes:
        qubit: The logical qubit associated with this wire start.
    """

    qubit: quariadne.circuit.LogicalQubit

    @property
    def label(self):
        return f"sq{self.qubit.index}"


@dataclasses.dataclass(frozen=True)
class WireEnd(ComputationalNode):
    """Represents the end of a DAG wire.

    Attributes:
        qubit: The logical qubit associated with this wire end.
    """

    qubit: quariadne.circuit.LogicalQubit

    @property
    def label(self):
        return f"eq{self.qubit.index}"


@dataclasses.dataclass(frozen=True)
class Gate(ComputationalNode):
    """Represents gate operations in the computational DAG.

    The gate_id parameter makes each gate unique to facilitate graph connections.

    Attributes:
        operation: The quantum operation containing name and participating qubits.
        gate_id: Unique identifier for the gate, used for DAG plotting and graph unwrapping
            to compare different node timesteps.
    """

    operation: quariadne.circuit.QuantumOperation

    # we don't want to include that field into representation because its ugly and uninformative
    gate_id: int = dataclasses.field(repr=False)

    @property
    def label(self):
        return self.operation.name


TRANSITION_QUBIT_FIELD = "qubit_index"


@dataclasses.dataclass
class Transition:
    """Represents a transition between two computational nodes in the DAG.

    Attributes:
        from_node: The source computational node.
        to_node: The destination computational node.
        underlying_qubit: The logical qubit associated with this transition.
    """

    from_node: ComputationalNode
    to_node: ComputationalNode
    underlying_qubit: quariadne.circuit.LogicalQubit

    def as_nx_edge(self):
        """Convert the transition to a NetworkX-compatible edge representation.

        Returns:
            NXEdge: Tuple containing the source node and target node.
        """
        nx_edge = (
            self.from_node,
            self.to_node,
        )
        return nx_edge


@dataclasses.dataclass
class ComputationalDAG:
    """Represents a computational circuit abstraction containing only routing-relevant objects.

    Attributes:
        nodes: List of computational nodes in the circuit.
        transitions: List of state transitions between nodes in the circuit.
    """

    nodes: typing.List[ComputationalNode]
    transitions: typing.List[Transition]

    @staticmethod
    def _convert_qiskit_dag_node(
        qiskit_dag_node: QiskitDAGNode,
    ) -> ComputationalNode:
        """Convert a Qiskit DAG node to the corresponding Quariadne node.

        Args:
            qiskit_dag_node: A Qiskit DAG node to convert to internal classes.

        Returns:
            The corresponding Quariadne computational node.

        Raises:
            TypeError: If an unexpected node type is encountered.
        """
        routing_node: ComputationalNode

        match type(qiskit_dag_node):
            case qiskit.dagcircuit.DAGOpNode:
                # this is a simple mapping of a qiskit operation node
                gate_name = qiskit_dag_node.op.name
                gate_wires = qiskit_dag_node.qargs
                gate_qubits_participating = tuple(
                    quariadne.circuit.LogicalQubit(wire._index) for wire in gate_wires
                )
                gate_operation = quariadne.circuit.QuantumOperation(
                    gate_name, gate_qubits_participating
                )
                gate_id = hash(qiskit_dag_node)
                routing_node = Gate(gate_operation, gate_id)
            case (
                (
                    qiskit.dagcircuit.DAGInNode | qiskit.dagcircuit.DAGOutNode
                ) as qiskit_node_type
            ):
                qubit_index = qiskit_dag_node.wire._index
                underlying_qubit = quariadne.circuit.LogicalQubit(qubit_index)
                # here we map the ending-starting parts of the dag to our nod eclasses
                if qiskit_node_type == qiskit.dagcircuit.DAGInNode:
                    routing_node = WireStart(underlying_qubit)
                else:
                    routing_node = WireEnd(underlying_qubit)
            case _:
                # for now we don't expect any other classes
                raise TypeError("Matching failed.")
        return routing_node

    @classmethod
    def _convert_qiskit_dag_edge(
        cls,
        qiskit_dag_edge: QiskitDAGEdge,
    ) -> Transition:
        """Convert a Qiskit DAG edge to a properly constructed transition object.

        Args:
            qiskit_dag_edge: A Qiskit DAG edge object tuple.

        Returns:
            The corresponding transition for the routing circuit.
        """
        # unpacking the meaningful objects
        in_node, out_node, wire = qiskit_dag_edge
        # mapping process here
        wire_index = wire._index
        underlying_qubit = quariadne.circuit.LogicalQubit(wire_index)
        in_routing_node = cls._convert_qiskit_dag_node(in_node)
        out_routing_node = cls._convert_qiskit_dag_node(out_node)
        # constructing the transition representation
        transition = Transition(in_routing_node, out_routing_node, underlying_qubit)
        return transition

    @classmethod
    def from_qiskit_dag(cls, qiskit_dag: qiskit.dagcircuit.DAGCircuit):
        """Convert a Qiskit DAG circuit to internal routing circuit representation.

        Args:
            qiskit_dag: A Qiskit DAG circuit object.

        Returns:
            ComputationalDAG: The converted routing circuit representation.
        """
        # obtaining generators for nodes and edges
        random_dag_nodes = qiskit_dag.nodes()
        random_dag_edges = qiskit_dag.edges()

        # preparing the arrays for nodes and transitions, then iterating through the corresponding qiskit generators
        # and filling the helper arrays
        circuit_nodes = []
        circuit_transitions = []
        for node in random_dag_nodes:
            circuit_node = cls._convert_qiskit_dag_node(node)
            circuit_nodes.append(circuit_node)

        for edge in random_dag_edges:
            circuit_transition = cls._convert_qiskit_dag_edge(edge)
            circuit_transitions.append(circuit_transition)

        # forming a resulting routing representation

        routing_circuit = cls(circuit_nodes, circuit_transitions)

        return routing_circuit

    def to_nx(self):
        """Converts the routing circuit into a directed graph in the NetworkX format.

        This method provides functionality to represent the graph in the NetworkX
        multiple directed graph format for interoperability with other libraries.

        Returns:
            nx.MultiDiGraph: Routing circuit as DAG in the NetworkX format.

        Raises:
            TypeError: if the provided underlying graph is not a DAG.
        """

        # creating a stub graph and filling it with the computational nodes
        routing_dag = nx.MultiDiGraph()
        routing_dag.add_nodes_from(self.nodes)

        # converting the transitions to edges
        transitions_as_edges = (
            transition.as_nx_edge() for transition in self.transitions
        )
        routing_dag.add_edges_from(transitions_as_edges)

        # checking that it is a dag
        if not nx.is_directed_acyclic_graph(routing_dag):
            raise TypeError("Graph should conform to DAG!")

        return routing_dag

    @staticmethod
    def _generate_topological_layout(routing_dag: nx.MultiDiGraph):
        """Generate a topological layout for the given DAG.

        Args:
            routing_dag: A NetworkX MultiDiGraph representing the DAG.

        Returns:
            dict: A multipartite layout dictionary mapping nodes to positions.
        """

        # first we generate a layer spread by topological sorting
        for layer, nodes in enumerate(nx.topological_generations(routing_dag)):
            for node in nodes:
                routing_dag.nodes[node]["layer"] = layer

        # Compute the multipartite_layout using the "layer" node attribute
        multipartite_layout = nx.multipartite_layout(routing_dag, subset_key="layer")
        return multipartite_layout

    def _generate_edge_labels_map(self):
        """Generate a mapper for edge plotting in NetworkX style.

        Returns:
            dict: Mapping from NetworkX edge to edge label string.
        """
        label_by_edge = {}
        for transition in self.transitions:
            edge = transition.as_nx_edge()
            edge_label = f"q{transition.underlying_qubit.index}"
            label_by_edge[edge] = edge_label

        return label_by_edge

    def _generate_node_labels_map(self):
        """Generate a mapper for node plotting in NetworkX style.

        Returns:
            dict: Mapping from nodes to their labels.
        """
        label_by_node = {}
        for node in self.nodes:
            label_by_node[node] = node.label
        return label_by_node

    def to_abstract_quantum_circuit(self) -> "quariadne.circuit.AbstractQuantumCircuit":
        """Convert the computational DAG to an AbstractQuantumCircuit.

        Filters out WireStart and WireEnd nodes, keeping only Gate nodes,
        and uses topological sorting to generate the chronological sequence of operations.

        Returns:
            AbstractQuantumCircuit: The converted circuit with operations in topological order.
        """
        nx_dag = self.to_nx()

        qubits = []
        for node in self.nodes:
            if isinstance(node, WireStart):
                qubits.append(node.qubit)

        operations = []
        for node in nx.topological_sort(nx_dag):
            if isinstance(node, Gate):
                operations.append(node.operation)

        abstract_circuit = quariadne.circuit.AbstractQuantumCircuit(
            operations, tuple(qubits)
        )
        return abstract_circuit

    def plot_dag(self):
        """Visualise the DAG corresponding to the routed circuit.

        Returns:
            tuple: A tuple containing the matplotlib figure and axis objects.
        """

        # generating helper topology, and label mappers
        routing_dag = self.to_nx()
        topological_layout = self._generate_topological_layout(routing_dag)
        label_by_node = self._generate_node_labels_map()
        label_by_edge = self._generate_edge_labels_map()

        # actual drawing of the graph happens
        dag_fig, dag_ax = plt.subplots()
        nx.draw_networkx(
            routing_dag, pos=topological_layout, ax=dag_ax, labels=label_by_node
        )
        nx.draw_networkx_edge_labels(
            routing_dag, pos=topological_layout, edge_labels=label_by_edge, ax=dag_ax
        )
        dag_ax.set_title(DAG_VISUALISATION_TITLE)
        dag_fig.tight_layout()

        return dag_fig, dag_ax
