import dataclasses
import typing


@dataclasses.dataclass(frozen=True)
class PhysicalQubit:
    """This class represents physical qubit, which is present in the coupling schemes."""

    index: int


@dataclasses.dataclass(frozen=True)
class LogicalQubit:
    """This class represents physical qubit, which is present in the coupling schemes."""

    index: int


@dataclasses.dataclass(frozen=True)
class QuantumOperation:
    """This is a class, which encapsulates the resolved quantum operation
    (think of it in terms of QCM), providing necessary information about it.

    Attributes:
        name: The operation name, usually something simple
        qubits_participating: The participating logical qubits
    """

    name: str
    qubits_participating: typing.Tuple[LogicalQubit, ...]


@dataclasses.dataclass()
class AbstractQuantumCircuit:
    """This is a main class of all the circuits, being a direct representation of the
    quantum circuit model.

    Attributes:
        operations: A chronologically ordered list of quantum operations in the circuit
        qubits: All the logical qubits, which participate in the circuit

    """

    operations: typing.List[QuantumOperation]
    qubits: typing.List[LogicalQubit]
