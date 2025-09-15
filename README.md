# Quariadne Qiskit Plugins

This README provides basic usage instructions for the Quariadne MILP-based layout and routing plugins for Qiskit transpilation.

## Installation

Install Quariadne using pip:

```bash
pip install quariadne
```

The plugins are automatically registered as Qiskit transpiler entry points upon installation.

## Plugin Overview

Quariadne provides two transpiler plugins:

- **Layout Plugin**: `quariadne_milp` - MILP-based initial qubit placement
- **Routing Plugin**: `quariadne_milp` - MILP-based SWAP insertion for circuit routing

Both plugins are registered as entry points in `pyproject.toml` and **must be used together** - they do not work when separated.

## Basic Usage

### Using with Preset Pass Manager

The plugins must be used together with Qiskit's `generate_preset_pass_manager()`:

```python
import qiskit
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime import SamplerV2 as Sampler


# Create a quantum circuit
circuit = qiskit.QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.cx(0, 2) # NOT POSSIBLE ON THE CHOSEN BACKEND!


# Set up backend
backend = FakeManilaV2()

# Generate pass manager with BOTH Quariadne plugins
pass_manager = qiskit.transpiler.generate_preset_pass_manager(
    backend=backend,
    optimization_level=3,
    layout_method="quariadne_milp",
    routing_method="quariadne_milp"  # Both must be used together
)

# Transpile the circuit
transpiled_circuit = pass_manager.run(circuit)

# Proper testing cycle with real sampler
with Session(backend=backend) as session:
    sampler = Sampler(mode=session)
    job = sampler.run([transpiled_circuit])
    result = job.result()[0]
    print(result)

```

## Important Limitations

### Circuit Size Constraints
For optimal performance, test with circuits that have:
- **Less than 50 gates**
- **Less than 5 qubits**

Larger circuits may experience significant performance degradation due to MILP complexity.

### Plugin Dependencies
**Critical**: The layout and routing plugins are interdependent and **cannot be used separately**. Using only one plugin will result in failure.


## Example Workflow

A complete example is available in `notebook/Testing.ipynb`, demonstrating:

1. Circuit creation with random Clifford gates (within size limits)
2. Backend setup using IBM's Manila fake provider  
3. Transpilation using **both** Quariadne MILP plugins together
4. Proper validation using real sampler execution

## Technical Details

- **Layout Plugin**: Converts Qiskit DAG to Quariadne's internal representation, solves MILP optimisation, and returns optimal qubit layout
- **Routing Plugin**: Uses MILP-determined SWAP operations from the layout stage and applies them layer-by-layer to the circuit

The routing plugin depends on results from the layout plugin, making them inseparable.

## Requirements

- Python >=3.13
- Qiskit >=2.0.1
- Backend with defined coupling map
- **Both plugins must be used together in the same pass manager**
- Test circuits should be small (< 50 gates, < 5 qubits)
- Use real samplers for proper testing validation

For more details on Qiskit transpiler plugins, refer to the [Qiskit Transpiler Documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/transpiler).



IMPORTANT DISCLAIMER: This code was written by me, with a serious help of Claude Code. Every bit of code was verified, but still there are risks. Use at your own risk. 