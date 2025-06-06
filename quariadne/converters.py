import typing
import qiskit.dagcircuit
import quariadne.circuit

type ConvertableQiskitNode = typing.Union[
    qiskit.dagcircuit.DAGOpNode,
    qiskit.dagcircuit.DAGInNode,
    qiskit.dagcircuit.DAGOutNode,
]


def convert_qiskit_node(qiskit_dag_node: ConvertableQiskitNode):
    """
    This function converts a qiskit dag node (there are three types of them until now, see :type ConvertableQiskitNode:)
    to the corresponding quariadne nodes.


    :param qiskit_dag_node: a node, which we expect to convert to internal classes
    :type qiskit_dag_node: ConvertableQiskitNode
    :return: returns the corresponding ariadne node
    :rtype: quariadne.circuit.RoutingNode
    """
    routing_node: quariadne.circuit.RoutingNode

    match type(qiskit_dag_node):
        case qiskit.dagcircuit.DAGOpNode:
            # this is a simple mapping of a qiskit operation node
            gate_name = qiskit_dag_node.op.name
            gate_num_qubits = qiskit_dag_node.num_qubits
            gate_id = hash(qiskit_dag_node)
            routing_node = quariadne.circuit.Gate(gate_num_qubits, gate_name, gate_id)
        case (
            qiskit.dagcircuit.DAGInNode
            | qiskit.dagcircuit.DAGOutNode as qiskit_node_type
        ):
            qubit_index = qiskit_dag_node.wire._index

            # here we map the ending-starting parts of the dag to our nod eclasses
            if qiskit_node_type == qiskit.dagcircuit.DAGInNode:
                routing_node = quariadne.circuit.WireStart(qubit_index)
            else:
                routing_node = quariadne.circuit.WireEnd(qubit_index)
        case _:
            # for now we don't expect any other classes
            raise TypeError("Matching failed.")
    return routing_node
