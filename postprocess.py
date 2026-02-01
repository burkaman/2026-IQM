import numpy as np
from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from iqm.qiskit_iqm.iqm_provider import IQMProvider
# from qrisp.interface import IQMBackend # Uncomment for real hardware

def create_noise_model(single_qubit_error=0.001, two_qubit_error=0.01, readout_error=0.02):
    """
    Creates a realistic noise model with gate and measurement errors.

    Args:
        single_qubit_error: Error probability for single-qubit gates (e.g., H)
        two_qubit_error: Error probability for two-qubit gates (e.g., CZ)
        readout_error: Measurement error probability
    """
    noise_model = NoiseModel()

    # Single-qubit gate errors (depolarizing)
    single_qubit_noise = depolarizing_error(single_qubit_error, 1)
    noise_model.add_all_qubit_quantum_error(single_qubit_noise, ['h'])

    # Two-qubit gate errors (depolarizing)
    two_qubit_noise = depolarizing_error(two_qubit_error, 2)
    noise_model.add_all_qubit_quantum_error(two_qubit_noise, ['cz', 'cx'])

    # Readout errors
    # Probability of measuring 1 when state is 0, and measuring 0 when state is 1
    readout_probs = [[1 - readout_error, readout_error],
                     [readout_error, 1 - readout_error]]
    readout_noise = ReadoutError(readout_probs)
    noise_model.add_all_qubit_readout_error(readout_noise)

    return noise_model

def bit_at_qubit(bitstring, q, n):
    # Qiskit: leftmost bit is highest index
    return 1 if bitstring[n - 1 - q] == "1" else 0

def stabilizer_Ki(bitstring, i, n):
    # K_i = Z_{i-1} X_i Z_{i+1}
    parity = bit_at_qubit(bitstring, i, n)
    if i > 0:
        parity ^= bit_at_qubit(bitstring, i-1, n)
    if i < n-1:
        parity ^= bit_at_qubit(bitstring, i+1, n)
    return +1 if parity == 0 else -1

def apply_stabilizer_postselection(counts, n, check_indices):
    kept = {}
    total = sum(counts.values())
    rejected = 0

    for b, c in counts.items():
        if all(stabilizer_Ki(b, i, n) == +1 for i in check_indices):
            kept[b] = kept.get(b, 0) + c
        else:
            rejected += c

    return kept, rejected / total if total else 0.0

def run_cluster_witness(
    n,
    backend,
    shots=2000,
    checks_A=None,
    checks_B=None
):
    checks_A = checks_A or []
    checks_B = checks_B or []

    # --- Prepare linear cluster state ---
    qc_base = QuantumCircuit(n)
    qc_base.h(range(n))
    for i in range(n-1):
        qc_base.cz(i, i+1)

    # --- Setting A: measure X on even sites ---
    qc_A = qc_base.copy()
    for i in range(0, n, 2):
        qc_A.h(i)
    qc_A.measure_all()

    # --- Setting B: measure X on odd sites ---
    qc_B = qc_base.copy()
    for i in range(1, n, 2):
        qc_B.h(i)
    qc_B.measure_all()

    # --- Run circuits ---
    jobA = backend.run(transpile(qc_A, backend), shots=shots)
    jobB = backend.run(transpile(qc_B, backend), shots=shots)

    counts_A = jobA.result().get_counts()
    counts_B = jobB.result().get_counts()

    # --- Postselection (ONLY stabilizers!) ---
    counts_A, rejA = apply_stabilizer_postselection(counts_A, n, checks_A)
    counts_B, rejB = apply_stabilizer_postselection(counts_B, n, checks_B)

    # --- Compute Theorem-2 probabilities ---
    def prob_all(indices, counts):
        total = sum(counts.values())
        if total == 0:
            return 0.0
        good = 0
        for b, c in counts.items():
            if all(stabilizer_Ki(b, i, n) == +1 for i in indices):
                good += c
        return good / total

    even_indices = [i for i in range(0, n, 2) if i not in checks_A]
    odd_indices  = [i for i in range(1, n, 2) if i not in checks_B]

    P_A = prob_all(even_indices, counts_A)
    P_B = prob_all(odd_indices, counts_B)

    W = 3 - 2 * (P_A + P_B)

    return W, P_A, P_B, rejA, rejB

# ============================================================
# Noise model (same spirit as your original)
# ============================================================

def make_noise():
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ['h'])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ['cz'])
    noise.add_all_qubit_readout_error(
        ReadoutError([[0.998, 0.002], [0.002, 0.998]])
    )
    return noise

provider = IQMProvider("https://resonance.meetiqm.com", quantum_computer="emerald",
                        token="HW9Qd7JxtPsZiMcR5QAf3sWpxjen12AedmSCu9Jq4ZUBnBdnp9JzEIXjmrn2NWsY")
backend = provider.get_backend()

# ============================================================
# Run comparison
# ============================================================

print("\n=== Cluster witness with stabilizer postselection ===")

for n in [10, 12, 14, 16, 18]:
    Wp, _, _, rA, rB = run_cluster_witness(
        n,
        backend,
        checks_A=[2] if n > 3 else [],
        checks_B=[1] if n > 2 else []
    )

    print(
        f"N={n:2d} | "
        f"W(ps)={Wp:+.3f} | "
        f"rejA={rA:.2f}, rejB={rB:.2f}"
    )