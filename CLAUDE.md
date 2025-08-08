# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. This is an experimental validation, most of the code is still written by human developers.

## Project Overview

Quariadne is a Python package for designing optimized qubit routing algorithms for quantum circuits. The project focuses on transforming quantum circuits from backends like Qiskit into internal representations suitable for routing optimization, then converting results back to the original circuit format.

## Core Architecture

The codebase follows a layered architecture:

### Circuit Abstraction Layer (`quariadne/circuit.py`)
- **PhysicalQubit**: Represents physical qubits in coupling maps
- **LogicalQubit**: Represents logical qubits with conversion from Qiskit wires
- **QuantumOperation**: Encapsulates quantum operations with participating qubits
- **AbstractQuantumCircuit**: Main circuit representation with chronologically ordered operations (partially ordered to be precise)

### Computational Graph Layer (`quariadne/computational_graph.py`)
- **ComputationalDAG**: DAG representation of quantum circuits for routing analysis
- **ComputationalNode**: Base class for DAG nodes (WireStart, WireEnd, Gate)
- **Transition**: Represents state transitions between computational nodes
- Conversion utilities between Qiskit DAG and internal representation
- NetworkX integration (mostly serving as a backbone) for graph algorithms and visualization

### Routing Engine (`quariadne/milp_router.py`)
- **MilpRouter**: Mixed-Integer Linear Programming router for qubit mapping
- Handles dummy qubit addition for hardware compatibility
- Generates constraint matrices for optimization (logical/physical uniqueness)
- Uses scipy optimization for routing solutions

## Development Commands

### Environment Setup
```bash
# Install dependencies (uses uv package manager)
uv sync
```

### Code Quality
```bash
# Type checking
mypy quariadne/

# Linting and formatting
ruff check quariadne/
ruff format quariadne/
```

### Documentation
```bash
# Build documentation with MkDocs
mkdocs build

# Serve documentation locally
mkdocs serve
```

### Testing
The project uses Jupyter notebooks for experimentation (see `notebook/MILP.ipynb`) and development mostly. All new features are usually handled there.

## Key Dependencies

- **qiskit**: Quantum circuit framework and DAG operations
- **networkx**: Graph algorithms and DAG manipulation
- **scipy**: Optimization routines for MILP solving
- **matplotlib**: Circuit and DAG visualization
- **numpy**: Mathematical operations and array handling

## Development Notes

- The project uses Python 3.13+ with type hints and dataclasses
- Qiskit integration is central to the circuit conversion pipeline (for now)
- NetworkX graphs are used extensively for DAG operations
- MyPy configuration ignores missing Qiskit imports (for now, maybe will commit a stub)
- The routing process involves timestep-based qubit mapping with swap operations


## MILP Implementation Notes

The `MilpRouter` class implements Mixed-Integer Linear Programming for quantum circuit routing using scipy.optimize.milp. Key concepts:

### MILP Mathematical Formulation
- Standard form: minimize c^T x subject to A*x ≤ b, bounds, and integrality constraints
- A is the coefficient matrix (populated from zero stubs)
- x contains decision variables: mapping, gate execution, and qubit movement variables
- Coefficient matrices start as zero arrays and are populated with constraint coefficients

### Constraint Generation Pattern
1. Initialize coefficient matrix stub using `_initialize_coefficient_matrix()`
2. Populate coefficients through nested loops over timesteps, qubits, edges
3. Use intermediate variables for each logical step (following code clarity principles)
4. Use `np.ravel_multi_index()` to convert multidimensional indices to flat array indices
5. Apply padding offsets using pre-calculated `flat_*` attributes
6. Collect constraints in lists and concatenate using `np.concatenate()`
7. Return populated coefficient matrix for scipy LinearConstraint objects

### Variable Indexing
- Variables are flattened: [mapping_vars, gate_execution_vars, movement_vars]
- Padding offsets ensure correct indexing across variable types
- Pre-calculated flat shapes: `flat_mapping_variables_shape`, `flat_gate_execution_variables_shape`, `flat_qubit_movement_shape`

### Implemented Constraint Generators
- `generate_logical_uniqueness_constraint()`: Each logical qubit maps to exactly one physical qubit per timestep
- `generate_physical_uniqueness_constraint()`: Each physical qubit hosts exactly one logical qubit per timestep  
- `generate_gate_execution_constraint()`: Each gate executes on exactly one physical edge at its assigned timestep

## Note for Claude Code 

Please, use always the recent documentation version (e.g. Python 3.13 https://docs.python.org/3/library ) and try to reference it in your implemented solutions *often*.  
Fetch documentation very often, it is always useful for you. Also, a good idea that would be smart to analyse always what the author changes in the code suggested, and ask if something wasn't obivous. 
Always try to update this file, or suggest an edit, if you see that the operator does something what you didn't expect! 
Please, use *only* Oxford English spelling!

**Code Documentation:**
Comments should NEVER repeat what's already obvious from the code
Only add comments that provide additional context, business logic, or non-obvious information
Focus on WHY something is done, not WHAT is being done


Good coding!