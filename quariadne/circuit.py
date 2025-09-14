import dataclasses
import typing

import qiskit


@dataclasses.dataclass(frozen=True)
class PhysicalQubit:
    """This class represents physical qubit, which is present in the coupling schemes."""

    index: int


@dataclasses.dataclass(frozen=True)
class LogicalQubit:
    """This class represents logical qubit, which is present in the coupling schemes."""

    index: int

    @classmethod
    def from_qiskit_wire(cls, qiskit_wire: qiskit.circuit.Qubit) -> typing.Self:
        """This function takes a qiskit circuit qubit, and converts it to corresponding
        logical qubit.

        Args:
            qiskit_wire: a qiskit circuit wire

        Returns:
            logical qubit
        """
        qiskit_wire_index = qiskit_wire._index
        logical_qubit = cls(qiskit_wire_index)
        return logical_qubit

    @classmethod
    def from_qiskit_wires(
        cls, qiskit_wires: typing.List[qiskit.circuit.Qubit]
    ) -> typing.Tuple[typing.Self, ...]:
        """This function takes a set of qiskit wires, and converts it to corresponding
        logical qubit iterable.

        Args:
            qiskit_wires: a qiskit circuit wire

        Returns:
            logical qubits tuple
        """
        logical_qubits = tuple(
            cls.from_qiskit_wire(qiskit_wire) for qiskit_wire in qiskit_wires
        )
        return logical_qubits


@dataclasses.dataclass(frozen=True)
class PhysicalSwap:
    """Represents a SWAP operation between two physical qubits.

    The swap is order-independent: PhysicalSwap(q1, q2) == PhysicalSwap(q2, q1).
    Uses frozenset for unordered comparison and hashing.

    Attributes:
        first: First physical qubit in the swap
        second: Second physical qubit in the swap
    """

    first: PhysicalQubit
    second: PhysicalQubit

    def __eq__(self, other: object) -> bool:
        """Compare swaps as unordered pairs."""
        if not isinstance(other, PhysicalSwap):
            return NotImplemented
        return frozenset([self.first, self.second]) == frozenset(
            [other.first, other.second]
        )

    def __hash__(self) -> int:
        """Hash based on unordered pair."""
        return hash(frozenset([self.first, self.second]))


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

    @classmethod
    def from_qiskit_instruction(
        cls, qiskit_instruction: qiskit.circuit.CircuitInstruction
    ):
        """This function gets a qiskit instruction, and returns an inner QuantumOperation object.

        Args:
            qiskit_instruction: a qiskit instruction, which defines the quantum operation.

        Returns:
            the corresponding quantum operation
        """

        # decomposing qiskit instrutciton
        instruction_operation, instruction_qubits = (
            qiskit_instruction.operation,
            qiskit_instruction.qubits,
        )
        quantum_operation_name = instruction_operation.name
        quantum_operation_qubits = LogicalQubit.from_qiskit_wires(instruction_qubits)
        quantum_operation = QuantumOperation(
            quantum_operation_name, quantum_operation_qubits
        )
        return quantum_operation


@dataclasses.dataclass()
class AbstractQuantumCircuit:
    """This is a main class of all the circuits, being a direct representation of the
    quantum circuit model.

    Attributes:
        operations: A chronologically ordered list of quantum operations in the circuit
        qubits: All the logical qubits, which participate in the circuit

    """

    operations: typing.List[QuantumOperation]
    qubits: typing.Tuple[LogicalQubit, ...]

    @classmethod
    def from_qiskit_circuit(cls, qiskit_circuit: qiskit.QuantumCircuit) -> typing.Self:
        """This function generates an abstract quantum circuit from a qiskit circuit

        Args:
            qiskit_circuit: a quantum circuit qiskit object, which we want to convert

        Returns:
            an abstract quantum circuit object, equivalent to qiskit one
        """

        # first we extract the list of logical qubits in the circuit
        computational_qubits = LogicalQubit.from_qiskit_wires(qiskit_circuit.qubits)
        operations = []

        # heret then we one by one extract the topologically sorted operations in the circuit
        for instruction in qiskit_circuit.data:
            # deconstructing the qiskit instruction
            instruction_operation, instruction_qubits = (
                instruction.operation,
                instruction.qubits,
            )
            quantum_operation_name = instruction_operation.name

            # creating the list of logical qubits
            quantum_operation_qubits = LogicalQubit.from_qiskit_wires(
                instruction_qubits
            )

            # constructing the corresponding operation
            quantum_operation = QuantumOperation(
                quantum_operation_name, quantum_operation_qubits
            )
            operations.append(quantum_operation)

        corresponding_abstract_circuit = cls(operations, computational_qubits)
        return corresponding_abstract_circuit
