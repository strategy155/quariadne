import qiskit.dagcircuit



def convert_qiskit_node(qiskit_dag_node):
    if isinstance(qiskit_dag_node, qiskit.dagcircuit.DAGOpNode):
        gate_name = qiskit_dag_node.op.name
        gate_num_qubits = qiskit_dag_node.num_qubits
        gate_id = hash(qiskit_dag_node)
        routing_node = Gate(gate_num_qubits, gate_name, gate_id)
    elif isinstance(qiskit_dag_node, (qiskit.dagcircuit.DAGInNode, qiskit.dagcircuit.DAGOutNode)):
        qubit_index = qiskit_dag_node.wire._index
        if isinstance(qiskit_dag_node, qiskit.dagcircuit.DAGInNode):
            qubit_is_start = True
        elif isinstance(qiskit_dag_node, qiskit.dagcircuit.DAGOutNode):
            qubit_is_start = False
        else:
            raise NotImplementedError("Wrong node types!")
        routing_node = Qubit(qubit_is_start, qubit_index)
    return routing_node