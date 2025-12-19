import random

import qiskit.circuit.random
import qiskit.transpiler
import qiskit_ibm_runtime.fake_provider
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime import SamplerV2 as Sampler

QUBIT_COUNT = 5
NUM_GATES = 20
SEED = 22363

random.seed(SEED)

MAX_OPERANDS = 2

STANDARD_GATE_SET = ["cx", "h"]


random_broken_circuit = qiskit.circuit.random.random_clifford_circuit(
    QUBIT_COUNT, NUM_GATES, STANDARD_GATE_SET, seed=SEED
)

random_broken_circuit.draw("mpl")

manila_backend = qiskit_ibm_runtime.fake_provider.FakeManilaV2()
coupling_graph = manila_backend.coupling_map.graph


basis_passmanager = qiskit.transpiler.generate_preset_pass_manager(
    backend=manila_backend,
    optimization_level=0,
    routing_method="quariadne_lpe",
    layout_method="quariadne_lpe",
)

physical = basis_passmanager.run(random_broken_circuit)
physical.draw("mpl")

with Session(backend=manila_backend) as session:
    # Submit a request to the Sampler primitive within the session.
    sampler = Sampler(mode=session)
    job = sampler.run([physical])
    pub_result = job.result()[0]
    print(pub_result)
