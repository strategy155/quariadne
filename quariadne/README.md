The schematics inside the quaridne is like so --- we have a set of logical qubits which keep track of the states of the qubits.

NB!: 

The interesting addition for the MILP itself would be to add something like an einsum interface,
which will allow to describe the operations of the constraints in a simple manner, and in turn generate the constraints.


So appranetly, the LP High Level router will not be a wrapper, but just a child of the Milp router class. Maybe even we will have an abstract class for that, for noow im' not SURE.

It will initialise the Milp router first with the continuous integrality and no other constraints --- it will be a bootstrap phase. 
From this run we will get the initial mapping, by the algorithm, which is shown in LP. Having an initial mapping, we provide it 
in a next phase, to get the next mapping. Then, when we have two mappings ready, we go to the loop, which will be the iterative one, until we'll be at the point when the amount of operations will be equal to 2 (2 last mappings to be fixed). In the iterative phase, we get the swap sequences by summing all the probabilities  (again you can see that in the notebook), and get a third mapping (the next one in sequence). Then we remove the first mapping, and go to the next iteration, until tehre will be no third mapping (because the circuit will be exhausted). That concludes the router. 
In the end we will have a sequence of mappings, and sequence of swaps. That is it. 