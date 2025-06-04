import typing
import qiskit.dagcircuit
import src.circuit

type ConvertableQiskitNode = typing.Union[
    qiskit.dagcircuit.DAGOpNode,
    qiskit.dagcircuit.DAGInNode,
    qiskit.dagcircuit.DAGOutNode,
]


def convert_qiskit_node(qiskit_dag_node: ConvertableQiskitNode):
    """
    This function converts a qiskit dag node (there are three types of them until now, see :type ConvertableQiskitNode:)
    to the corresponding ariadne nodes.


    :param qiskit_dag_node: a node, which we expect to convert to internal classes
    :type qiskit_dag_node: ConvertableQiskitNode
    :return: returns the corresponding ariadne node
    :rtype: ariadne.circuit.RoutingNode
    """

    # THis is a simple type-mapper to different inner ariadne type
    if isinstance(qiskit_dag_node, qiskit.dagcircuit.DAGOpNode):
        gate_name = qiskit_dag_node.op.name
        gate_num_qubits = qiskit_dag_node.num_qubits
        gate_id = hash(qiskit_dag_node)
        routing_node = src.circuit.Gate(gate_num_qubits, gate_name, gate_id)
    elif isinstance(
        qiskit_dag_node, (qiskit.dagcircuit.DAGInNode, qiskit.dagcircuit.DAGOutNode)
    ):
        qubit_index = qiskit_dag_node.wire._index
        if isinstance(qiskit_dag_node, qiskit.dagcircuit.DAGInNode):
            qubit_is_start = True
        elif isinstance(qiskit_dag_node, qiskit.dagcircuit.DAGOutNode):
            qubit_is_start = False
        else:
            raise NotImplementedError("Wrong node types!")
        routing_node = src.circuit.LogicalQubit(qubit_is_start, qubit_index)
    else:
        raise TypeError("Unsupported types")
    return routing_node
