Ariadne is a package for designing an engine for the purposes of the optimised qubit routing algorithms. 

For now the basic flow is considered being like this (experiments still ongoing):

1. We plug in some circuit backend, like qiskit. 
2. We transform it to internal routing-specific representation.
3. We then run the specific algorithms on the internal representation.
4. Then we convert the routed results back to the circuit backend.

