import numpy as np
from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt
from iqm.qiskit_iqm.iqm_provider import IQMProvider

def run_ghz_witness_theorem_1(n_qubits, backend, shots=2000):
    """
    Implements the GHZ State Witness from Toth & Guhne Theorem 1.
    
    Stabilizers (Eq. 5):
        S_1 = X^{⊗N}  (all X)
        S_k = Z_{k-1} Z_k  for k = 2, ..., N
    
    Witness (Eq. 6):
        W = 3I - 2 * [ (S_1 + I)/2  +  prod_{k=2}^{N} (S_k + I)/2 ]
    
    The two terms in brackets are projectors:
        P_A = (S_1 + I)/2  -->  projects onto S_1 = +1 eigenspace
        P_B = prod_{k=2}^{N} (S_k + I)/2  -->  projects onto simultaneous +1 eigenspace of all S_k
    
    Since S_1 and each S_k have eigenvalues ±1:
        P_A = 1 if S_1 = +1, else 0
        P_B = 1 if ALL S_k = +1 simultaneously, else 0
    
    So expectation values become probabilities:
        <P_A> = Prob(all qubits have even parity in X basis)
        <P_B> = Prob(every adjacent pair has even parity in Z basis)
    
    W = 3 - 2*(P_A + P_B)
    Target: W < 0 implies Genuine Multipartite Entanglement.
    Perfect GHZ: W = 3 - 2*(1 + 1) = -1
    """
    
    # --- 1. PREPARE THE GHZ STATE ---
    # |GHZ_N> = (|00...0> + |11...1>) / sqrt(2)
    qc_base = QuantumCircuit(n_qubits)
    qc_base.h(0)                          # Put qubit 0 into |+>
    for i in range(n_qubits - 1):
        qc_base.cx(i, i + 1)             # CNOT cascade: propagates superposition

    # --- 2. DEFINE THE TWO MEASUREMENT SETTINGS (Fig 1b) ---
    
    # Setting A: Measure S_1 = X^{⊗N}
    # To measure X on every qubit, apply H to all qubits before measurement.
    # The global parity of the resulting bitstring gives the eigenvalue of S_1.
    # S_1 = +1 iff parity is even (even number of 1s).
    qc_A = qc_base.copy()
    qc_A.h(range(n_qubits))              # Rotate all qubits: X eigenbasis -> Z eigenbasis
    qc_A.measure_all()
    
    # Setting B: Measure S_k = Z_{k-1} Z_k for k = 2, ..., N
    # Z is the computational basis, so no rotation needed.
    # Each S_k = +1 iff qubits (k-1) and k have the same value (even parity on that pair).
    # P_B = 1 iff ALL adjacent pairs agree simultaneously.
    qc_B = qc_base.copy()
    # No basis change needed -- already measuring in Z basis
    qc_B.measure_all()
    
    # --- 3. RUN EXPERIMENT ---
    print(f"Running GHZ Witness for N={n_qubits}...")
    job_A = backend.run(transpile(qc_A, backend), shots=shots)
    job_B = backend.run(transpile(qc_B, backend), shots=shots)
    counts_A = job_A.result().get_counts()
    counts_B = job_B.result().get_counts()
    
    # --- 4. CALCULATE PROBABILITIES ---
    
    # P_A: Prob(S_1 = +1) = Prob(even global parity in X-basis measurement)
    # S_1 = X^{⊗N} has eigenvalue (-1)^{number of 1s in bitstring after H rotation}
    # So S_1 = +1 iff total number of 1s is even.
    success_count_A = 0
    for bitstring, count in counts_A.items():
        parity = bitstring.count('1') % 2   # Global parity
        if parity == 0:                      # Even parity => S_1 = +1
            success_count_A += count
    prob_A = success_count_A / shots

    # P_B: Prob(all S_k = +1 simultaneously) = Prob(all adjacent pairs agree in Z basis)
    # S_k = Z_{k-1} Z_k = +1 iff qubit (k-1) and qubit k have the same measurement outcome.
    # P_B = 1 iff the entire bitstring is all 0s or all 1s (the two GHZ components).
    # Note: Qiskit bitstrings are reversed -- bit index 0 in string is qubit N-1.
    success_count_B = 0
    for bitstring, count in counts_B.items():
        all_pairs_agree = True
        for k in range(1, n_qubits):        # S_k for k = 2, ..., N (0-indexed: k = 1, ..., N-1)
            # Qubit indices in Qiskit reversed bitstring:
            bit_k_minus_1 = bitstring[n_qubits - 1 - (k - 1)]
            bit_k        = bitstring[n_qubits - 1 - k]
            if bit_k_minus_1 != bit_k:       # Z_{k-1} Z_k = -1 if they differ
                all_pairs_agree = False
                break
        if all_pairs_agree:
            success_count_B += count
    prob_B = success_count_B / shots

    # --- 5. COMPUTE WITNESS VALUE ---
    # W = 3 - 2 * (P_A + P_B)
    w_value = 3 - 2 * (prob_A + prob_B)
    
    return w_value, prob_A, prob_B

provider = IQMProvider("https://resonance.meetiqm.com", quantum_computer="emerald",
                        token="HW9Qd7JxtPsZiMcR5QAf3sWpxjen12AedmSCu9Jq4ZUBnBdnp9JzEIXjmrn2NWsY")
backend = provider.get_backend()
print(f"{'N':>4} | {'Witness W':>10} | {'P_A (X parity)':>14} | {'P_B (Z pairs)':>14} | {'GME?':>5}")
print("-" * 60)
for n in [14]:
    w, pA, pB = run_ghz_witness_theorem_1(n, backend, shots=4000)
    gme = "YES" if w < 0 else "no"
    print(f"{n:>4} | {w:>10.3f} | {pA:>14.4f} | {pB:>14.4f} | {gme:>5}")
    # Perfect state: P_A = 1, P_B = 1, W = -1
    # With noise: both probabilities decay, W rises toward +3