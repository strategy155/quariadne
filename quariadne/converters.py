import typing
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


def convert_qiskit_dag_node(
    qiskit_dag_node: QiskitDAGNode,
) -> quariadne.circuit.RoutingNode:
    """
    This function converts a qiskit dag node (there are three types of them until now, see :type QiskitDAGNode:)
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
            (
                qiskit.dagcircuit.DAGInNode | qiskit.dagcircuit.DAGOutNode
            ) as qiskit_node_type
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


def convert_qiskit_dag_edge(qiskit_dag_edge: QiskitDAGEdge):
    """
    This function gets a qiskit dag edge, and returns properly constructed transition object.
    :param qiskit_dag_edge: a qiskit dag edge object, which is a tuple of a special type.
    :return: returns the corresponding transition for the routing circuit
    :rtype: quariadne.circuit.Transition
    """
    # unpacking the meaningful objects
    in_node, out_node, wire = qiskit_dag_edge
    # mapping process here
    wire_index = wire._index
    in_routing_node = convert_qiskit_dag_node(in_node)
    out_routing_node = convert_qiskit_dag_node(out_node)
    # constructing the transition representation
    transition = quariadne.circuit.Transition(
        in_routing_node, out_routing_node, wire_index
    )
    return transition
