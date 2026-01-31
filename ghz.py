from iqm.qiskit_iqm.iqm_provider import IQMProvider
from qiskit import QuantumCircuit, transpile

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)
qc.measure_all()

provider = IQMProvider(
    "https://resonance.meetiqm.com",
    quantum_computer="sirius",
    token="",
)
backend = provider.get_backend()

qc_transpiled = transpile(qc, backend)
job = backend.run(qc_transpiled)

results = job.result()
print(results.get_counts())
