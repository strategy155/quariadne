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
    """
    This is a baseclass for all the nodes in the internal DAG representation of the routing circuit.
    """

    @property
    @abc.abstractmethod
    def label(self):
        """
        This method should return the label of the node,
        for plotting purposes
        """
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class WireStart(ComputationalNode):
    """
    This is a class, which represents the DAG wire start.
    """

    qubit: quariadne.circuit.LogicalQubit

    @property
    def label(self):
        return f"sq{self.qubit.index}"


@dataclasses.dataclass(frozen=True)
class WireEnd(ComputationalNode):
    """
    This is a class, which represents the end of the DAG wire
    """

    qubit: quariadne.circuit.LogicalQubit

    @property
    def label(self):
        return f"eq{self.qubit.index}"


@dataclasses.dataclass(frozen=True)
class Gate(ComputationalNode):
    """This is a class, representing the gate operations. It now has this parent_id parameter,
    which, for now, makes it unique (to make the connections in the graph easier.

    Args:
        qubits_participating: controls the amount of logical qubits participating in the gate
        name: the name of the operation

        gate_id: exclusively needed for the DAG plotting, and graph unwrapping, to compare different node-timesteps.

    """

    qubits_participating: typing.Tuple[quariadne.circuit.LogicalQubit, ...]
    name: str

    # we don't want to include that field into representation because its ugly and uninformative
    gate_id: int = dataclasses.field(repr=False)

    @property
    def label(self):
        return self.name


TRANSITION_QUBIT_FIELD = "qubit_index"


@dataclasses.dataclass
class Transition:
    """
    This class represents the transition between two computational nodes in the DAG.
    """

    from_node: ComputationalNode
    to_node: ComputationalNode
    underlying_qubit: quariadne.circuit.LogicalQubit

    def as_nx_edge(self):
        """
        Converts the current edge to a NetworkX-compatible edge representation.

        Returns the edge as a tuple containing the source node and target node,
        which can be used directly with NetworkX graphs.

        :return nx_edge: Edge as a tuple containing the source node and target node.
        :rtype NXEdge:
        """
        nx_edge = (
            self.from_node,
            self.to_node,
        )
        return nx_edge


@dataclasses.dataclass
class ComputationalDAG:
    """This class represents a computational circuit abstraction,
     containing only relevant objects for routing

    Args:
        nodes: list of computational nodes in the circuit
        transitions: list of state transitions in the circuit between the nodes


    """

    nodes: typing.List[ComputationalNode]
    transitions: typing.List[Transition]

    @staticmethod
    def _convert_qiskit_dag_node(
        qiskit_dag_node: QiskitDAGNode,
    ) -> ComputationalNode:
        """This function converts a qiskit dag node (there are three types of them until now, see :type QiskitDAGNode:)
        to the corresponding quariadne nodes.

        Args:
            qiskit_dag_node (QiskitDAGEdge): a node, which we expect to convert to internal classes
            current_timestep (int): the current timestep of the circuit with the converter

        Returns:
            quariadne.circuit.ComputationalNode: The function returns the corresponding quariadne node

        Raises:
            TypeError, if the function got an unexpected type of node
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
                gate_id = hash(qiskit_dag_node)
                routing_node = Gate(gate_qubits_participating, gate_name, gate_id)
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
        """this function gets a qiskit dag edge, and returns properly constructed transition object.

         Args:
             qiskit_dag_edge (QiskitDAGEdge): a qiskit dag edge object, which is a tuple of a special type.

        Returns:
            It returns the corresponding transition for the routing circuit
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
        """This function takes qiskit dag circuit as input,
         and converts it to internal routing circuit representation.

        Args:
            qiskit_dag: a qiskit dag circuit object, which is easily obtained from qiskit circuit.


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

        routing_circuit = ComputationalDAG(circuit_nodes, circuit_transitions)

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
        """This function gets a DAG and then generates a topological layout for it."""

        # first we generate a layer spread by topological sorting
        for layer, nodes in enumerate(nx.topological_generations(routing_dag)):
            for node in nodes:
                routing_dag.nodes[node]["layer"] = layer

        # Compute the multipartite_layout using the "layer" node attribute
        multipartite_layout = nx.multipartite_layout(routing_dag, subset_key="layer")
        return multipartite_layout

    def _generate_edge_labels_map(self):
        """This function generates a mapper for edges plotting (networkx style)

        Returns:
            dict: a map of nx edge to edge label string
        """
        label_by_edge = {}
        for transition in self.transitions:
            edge = transition.as_nx_edge()
            edge_label = f"q{transition.underlying_qubit.index}"
            label_by_edge[edge] = edge_label

        return label_by_edge

    def _generate_node_labels_map(self):
        """This function generates a mapper for nodes plotting (networkx style)."""
        label_by_node = {}
        for node in self.nodes:
            label_by_node[node] = node.label
        return label_by_node

    def plot_dag(self):
        """This function visualises the DAG corresponding to the routed circuit"""

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
