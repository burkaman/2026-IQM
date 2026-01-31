from iqm.qiskit_iqm.iqm_provider import IQMProvider
from qiskit import QuantumCircuit, transpile


# 1. CREATE BASE CIRCUIT
def create_cluster(n=4):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    # Ring: 0-1-2-3-0
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        qc.cz(i, j)
    return qc


# 2. MEASURE PAULI STRINGS
def measure_pauli_string(pauli_bases, shots=2048):
    qc = create_cluster(len(pauli_bases))

    # Rotate to measurement basis
    for i, basis in enumerate(pauli_bases):
        if basis == "X":
            qc.h(i)
        elif basis == "Y":
            qc.sdg(i)
            qc.h(i)

    qc.measure_all()

    # Run and get counts
    provider = IQMProvider(
        "https://resonance.meetiqm.com",
        quantum_computer="sirius",
        token="",
    )
    backend = provider.get_backend()
    qc_transpiled = transpile(qc, backend)
    job = backend.run(qc_transpiled, shots=shots)
    results = job.result()
    counts = results.get_counts()

    # Compute expectation
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring[::-1]
        parity = sum(
            1
            for i, (bit, b) in enumerate(zip(bits, pauli_bases))
            if b != "I" and bit == "1"
        )
        exp += (1 if parity % 2 == 0 else -1) * count / shots
    return exp


# 3. RUN THE 5 MEASUREMENT CIRCUITS
stabilizers = [
    ["X", "Z", "I", "Z"],
    ["Z", "X", "Z", "I"],
    ["I", "Z", "X", "Z"],
    ["Z", "I", "Z", "X"],
]

sum_stabs = sum(measure_pauli_string(s, 2048) for s in stabilizers)
all_x = measure_pauli_string(["X", "X", "X", "X"], 2048)

# 4. COMPUTE WITNESS
W = 2.0 - 0.5 * (sum_stabs + all_x)

# 5. DETECT
if W < -0.05:
    print(f"✓ ENTANGLED! W = {W:.4f}")
else:
    print(f"✗ Not detected. W = {W:.4f}")
