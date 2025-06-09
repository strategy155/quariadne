import dataclasses
import typing
import networkx as nx
import abc
import matplotlib.pyplot as plt


type NXEdge = typing.Tuple[RoutingNode, RoutingNode]

DAG_VISUALISATION_TITLE = "Visualised DAG"


@dataclasses.dataclass(frozen=True)
class RoutingNode(abc.ABC):
    """
    This is a baseclass for all the nodes in the internal representation of the routing circuit.
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
class WireStart(RoutingNode):
    """
    This is a class, which represents the DAG wire start.
    """

    index: int

    @property
    def label(self):
        return f"sq{self.index}"


@dataclasses.dataclass(frozen=True)
class WireEnd(RoutingNode):
    """
    This is a class, which represents the end of the DAG wire
    """

    index: int

    @property
    def label(self):
        return f"eq{self.index}"


@dataclasses.dataclass(frozen=True)
class Gate(RoutingNode):
    """
    This is a class, representing the gate operations. It now has this parent_id parameter,
    which, for now, makes it unique (to make the connections in the graph easier.
    :param qubits_participating: controls the amount of logical qubits participating in the gate
    :param name: the name of the operation
    :param parent_id: for now this is a relict from the qiskit library, specifying what operation hash was the source of this gate.

    """

    qubits_participating: int
    name: str
    # we don't want to include that field into representation because its ugly and uninformative
    parent_id: int = dataclasses.field(repr=False)

    @property
    def label(self):
        return self.name


TRANSITION_QUBIT_FIELD = "qubit_index"


@dataclasses.dataclass
class Transition:
    """
    This class represents the transition between two computational nodes in the DAG.
    """

    from_node: RoutingNode
    to_node: RoutingNode
    qubit_index: int

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
class RoutingCircuit:
    """
    This class represents a circuit abstraction, containing only relevant objects for routing


    """

    nodes: typing.List[RoutingNode]
    transitions: typing.List[Transition]

    def to_nx(self):
        """
        Converts the routing circuit into a directed graph in the NetworkX format.

        This method provides functionality to represent the graph in the NetworkX
        multiple directed graph format for interoperability with other libraries.

        :return nx_multidigraph: Routing circuit as DAG in the NetworkX format.
        :rtype nx.MultiDiGraph:
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
        """
        This function gets a DAG and then generates a topological layout for it.
        """

        # first we generate a layer spread by topological sorting
        for layer, nodes in enumerate(nx.topological_generations(routing_dag)):
            for node in nodes:
                routing_dag.nodes[node]["layer"] = layer

        # Compute the multipartite_layout using the "layer" node attribute
        multipartite_layout = nx.multipartite_layout(routing_dag, subset_key="layer")
        return multipartite_layout

    def _generate_edge_labels_map(self):
        """
        This function generates a mapper for edges plotting (networkx style)
        :return label_by_edge: a map of nx edge to edge label
        :rtype: dict
        """
        label_by_edge = {}
        for transition in self.transitions:
            edge = transition.as_nx_edge()
            edge_label = f"q{transition.qubit_index}"
            label_by_edge[edge] = edge_label

        return label_by_edge

    def _generate_node_labels_map(self):
        """
        This function generates a mapper for nodes plotting (networkx style).
        """
        label_by_node = {}
        for node in self.nodes:
            label_by_node[node] = node.label
        return label_by_node

    def plot_dag(self):
        """
        This function visualises the DAG corresponding to the routed circuit
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
