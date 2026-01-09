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


@dataclasses.dataclass(frozen=True, eq=False)
class QuantumOperation:
    """This is a class, which encapsulates the resolved quantum operation
    (think of it in terms of QCM), providing necessary information about it.
    They are always unique, for making the quantum circuit abstraction valid.

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

    def get_two_qubit_operations(self) -> typing.List[QuantumOperation]:
        """Extract two-qubit operations from the circuit.

        Filters the circuit operations to include only two-qubit gates, which are the
        ones that require routing due to coupling map constraints. Single-qubit operations
        can be executed on any physical qubit without routing considerations.

        Returns:
            List of QuantumOperation objects that involve exactly two qubits.

        Raises:
            TypeError: If any operation involves more than two qubits.
        """
        two_qubit_gate_operations = []
        for operation in self.operations:
            if len(operation.qubits_participating) == 2:
                two_qubit_gate_operations.append(operation)
            elif len(operation.qubits_participating) > 2:
                raise TypeError("We got much more qubits that we want!")

        return two_qubit_gate_operations

    def get_slice(self, start: int, end: int | None = None) -> "AbstractQuantumCircuit":
        """Create a new circuit with operations sliced from start to end index.

        Returns a new AbstractQuantumCircuit instance with operations sliced from the
        specified start index to the end index (or to the end if not specified).
        The qubits remain unchanged, preserving immutability.

        Args:
            start: Starting index for the slice (inclusive)
            end: Ending index for the slice (exclusive). If None, slices to the end.

        Returns:
            New AbstractQuantumCircuit with the sliced operations.

        Raises:
            IndexError: If start index is out of range.
        """
        if start < 0 or start > len(self.operations):
            raise IndexError(
                f"Start index {start} out of range for circuit with {len(self.operations)} operations"
            )

        sliced_operations = self.operations[start:end]
        return AbstractQuantumCircuit(sliced_operations, self.qubits)

    def shift_first_operation(self) -> None:
        """Remove the first operation from the circuit in-place.

        Modifies the operations list by removing the first element.
        This is useful for iterative processing where operations are consumed one by one.

        Raises:
            IndexError: If the circuit has no operations to shift.
        """
        if not self.operations:
            raise IndexError("Cannot shift from empty circuit")

        self.operations.pop(0)

    def to_qiskit(self) -> qiskit.QuantumCircuit:
        """Convert this AbstractQuantumCircuit to a Qiskit QuantumCircuit.

        Creates an equivalent Qiskit QuantumCircuit with the same operations.
        Uses the operation name to dispatch to the correct Qiskit gate method.

        Returns:
            Equivalent Qiskit QuantumCircuit.

        Reference:
            - Qiskit QuantumCircuit: https://docs.quantum.ibm.com/api/qiskit/circuit
        """
        num_qubits = len(self.qubits)
        qiskit_circuit = qiskit.QuantumCircuit(num_qubits)

        for operation in self.operations:
            qubit_indices = [q.index for q in operation.qubits_participating]
            gate_method = getattr(qiskit_circuit, operation.name, None)
            if gate_method is not None:
                gate_method(*qubit_indices)

        return qiskit_circuit
