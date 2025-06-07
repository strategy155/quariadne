import dataclasses
import typing
import networkx as nx

type RoutingNode = WireStart | WireEnd | Gate
type NXEdge = typing.Tuple[RoutingNode, RoutingNode]


@dataclasses.dataclass(frozen=True)
class WireStart:
    """
    This is a class, which represents the DAG wire start.
    """

    index: int


@dataclasses.dataclass(frozen=True)
class WireEnd:
    """ "
    This is a class, which represents the end of the DAG"""

    index: int


@dataclasses.dataclass(frozen=True)
class Gate:
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

    def as_nx_digraph(self):
        """
        Converts the routing circuit into a directed graph in the NetworkX format.

        This method provides functionality to represent the graph in the NetworkX
        directed graph format for interoperability with other libraries.

        :return nx_digraph: Routing circuit as DAG in the NetworkX format.
        :rtype nx.DiGraph:
        """

        # creating a stub graph and filling it with the computational nodes
        random_routing_dag = nx.DiGraph()
        random_routing_dag.add_nodes_from(self.nodes)

        # converting the transitions to edges
        transitions_as_edges = (
            transition.as_nx_edge() for transition in self.transitions
        )
        random_routing_dag.add_edges_from(transitions_as_edges)

        return random_routing_dag
