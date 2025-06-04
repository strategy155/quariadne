import dataclasses


type RoutingNode = LogicalQubit | Gate


@dataclasses.dataclass(frozen=True)
class LogicalQubit:
    """
    This is a class, which represents the logical qubit (or wire, in some terminology
    -- this terminology comes from the quantum circuit model).
    """

    is_start: bool
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
