import dataclasses
import typing

type RoutingNode = WireStart | WireEnd | Gate


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


@dataclasses.dataclass
class RoutingCircuit:
    """
    This class represents a circuit abstraction, containing only relevant objects for routing
    """

    nodes: typing.List[RoutingNode]
    transitions: typing.List[Transition]
