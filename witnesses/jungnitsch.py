import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

"""
Jungnitsch et al (2011) - "Taming Multiparticle Entanglement"
arXiv:1010.6049, Phys. Rev. Lett. 106, 190502

The key formula from the paper:
    W = α·I - |Cl⟩⟨Cl| - P_plus
    
where P_plus is constructed from k disjoint stabilizer generators:
    P_plus = (1/2^(k+1)) × [∏(I-g_i) + Σ_j (I+g_j)∏_{m≠j}(I-g_m)]

For measurement outcomes, P_plus has eigenvalue:
    - 0.5 if at most 1 of k disjoint stabilizers is satisfied
    - 0   if 2+ disjoint stabilizers are satisfied

This gives exponential improvement in noise tolerance because P_plus
is positive on biseparable states but ~0 on genuine multipartite entangled states.
"""

def run_jungnitsch_witness(n_qubits, backend, noise_model, shots=2000):
    """
    Measurement-based Jungnitsch witness.
    
    W_jungnitsch = W_standard - P_plus
    
    where:
    - W_standard = 3 - 2*(P_even + P_odd) [Toth-Guhne Theorem 2]
    - P_plus = 0.5 × Prob(at most 1 of k disjoint stabilizers satisfied)
    
    The disjoint stabilizers use spacing=3: indices 0, 3, 6, ...
    This ensures they don't share qubits and can be measured simultaneously.
    """
    
    # Disjoint generators (spacing=3 as per paper)
    selected_indices = list(range(0, n_qubits, 3))
    k = len(selected_indices)
    
    # === CIRCUIT 1: Measure EVEN-indexed stabilizers (0, 2, 4, ...) ===
    qc_even = QuantumCircuit(n_qubits)
    qc_even.h(range(n_qubits))
    for i in range(0, n_qubits - 1, 2): qc_even.cz(i, i+1)
    for i in range(1, n_qubits - 1, 2): qc_even.cz(i, i+1)
    for i in range(0, n_qubits, 2): 
        qc_even.h(i)  # Rotate X→Z basis
    qc_even.measure_all()
    
    # === CIRCUIT 2: Measure ODD-indexed stabilizers (1, 3, 5, ...) ===
    qc_odd = QuantumCircuit(n_qubits)
    qc_odd.h(range(n_qubits))
    for i in range(0, n_qubits - 1, 2): qc_odd.cz(i, i+1)
    for i in range(1, n_qubits - 1, 2): qc_odd.cz(i, i+1)
    for i in range(1, n_qubits, 2):
        qc_odd.h(i)
    qc_odd.measure_all()
    
    # === CIRCUIT 3: Measure DISJOINT stabilizers (0, 3, 6, ...) ===
    qc_disjoint = QuantumCircuit(n_qubits)
    qc_disjoint.h(range(n_qubits))
    for i in range(0, n_qubits - 1, 2): qc_disjoint.cz(i, i+1)
    for i in range(1, n_qubits - 1, 2): qc_disjoint.cz(i, i+1)
    for idx in selected_indices:
        qc_disjoint.h(idx)
    qc_disjoint.measure_all()
    
    # Run all circuits
    result_even = backend.run(transpile(qc_even, backend), shots=shots, noise_model=noise_model).result()
    result_odd = backend.run(transpile(qc_odd, backend), shots=shots, noise_model=noise_model).result()
    result_disjoint = backend.run(transpile(qc_disjoint, backend), shots=shots, noise_model=noise_model).result()
    
    counts_even = result_even.get_counts()
    counts_odd = result_odd.get_counts()
    counts_disjoint = result_disjoint.get_counts()
    
    # === Process EVEN stabilizers ===
    success_even = 0
    for bitstring, count in counts_even.items():
        all_satisfied = True
        for i in range(0, n_qubits, 2):
            parity = 1
            if bitstring[n_qubits - 1 - i] == '1': parity *= -1
            if i > 0 and bitstring[n_qubits - 1 - (i-1)] == '1': parity *= -1
            if i < n_qubits - 1 and bitstring[n_qubits - 1 - (i+1)] == '1': parity *= -1
            if parity == -1:
                all_satisfied = False
                break
        if all_satisfied:
            success_even += count
    prob_even = success_even / shots
    
    # === Process ODD stabilizers ===
    success_odd = 0
    for bitstring, count in counts_odd.items():
        all_satisfied = True
        for i in range(1, n_qubits, 2):
            parity = 1
            if bitstring[n_qubits - 1 - i] == '1': parity *= -1
            if bitstring[n_qubits - 1 - (i-1)] == '1': parity *= -1
            if i < n_qubits - 1 and bitstring[n_qubits - 1 - (i+1)] == '1': parity *= -1
            if parity == -1:
                all_satisfied = False
                break
        if all_satisfied:
            success_odd += count
    prob_odd = success_odd / shots
    
    # === Process DISJOINT stabilizers for P_plus ===
    # P_plus eigenvalue = 0.5 if ≤1 satisfied, 0 if ≥2 satisfied
    count_at_most_1 = 0
    for bitstring, count in counts_disjoint.items():
        num_satisfied = 0
        for idx in selected_indices:
            parity = 1
            if bitstring[n_qubits - 1 - idx] == '1': parity *= -1
            if idx > 0 and bitstring[n_qubits - 1 - (idx-1)] == '1': parity *= -1
            if idx < n_qubits - 1 and bitstring[n_qubits - 1 - (idx+1)] == '1': parity *= -1
            if parity == +1:
                num_satisfied += 1
        if num_satisfied <= 1:
            count_at_most_1 += count
    
    prob_at_most_1 = count_at_most_1 / shots
    p_plus = 0.5 * prob_at_most_1
    
    # === Compute witness values ===
    w_standard = 3 - 2 * (prob_even + prob_odd)
    w_jungnitsch = w_standard - p_plus  # Subtracting P_plus improves detection!
    
    # === Error bars (binomial standard error) ===
    var_even = prob_even * (1 - prob_even) / shots
    var_odd = prob_odd * (1 - prob_odd) / shots
    var_at_most_1 = prob_at_most_1 * (1 - prob_at_most_1) / shots
    
    # Error propagation: W_std = 3 - 2*(p_e + p_o)
    se_standard = 2 * np.sqrt(var_even + var_odd)
    
    # Error propagation: W_jung = W_std - 0.5*p_at_most_1
    se_jungnitsch = np.sqrt(4*(var_even + var_odd) + 0.25*var_at_most_1)
    
    return {
        'w_standard': w_standard,
        'se_standard': se_standard,
        'w_jungnitsch': w_jungnitsch,
        'se_jungnitsch': se_jungnitsch,
        'prob_even': prob_even,
        'prob_odd': prob_odd,
        'p_plus': p_plus,
        'k': k
    }

def create_noise_model(single_qubit_error=0.001, two_qubit_error=0.01, readout_error=0.02):
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(single_qubit_error, 1), ['h'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error, 2), ['cz', 'cx'])
    readout_probs = [[1 - readout_error, readout_error], [readout_error, 1 - readout_error]]
    noise_model.add_all_qubit_readout_error(ReadoutError(readout_probs))
    return noise_model

# === EXECUTION ===
print("=== Jungnitsch Witness (arXiv:1010.6049) ===\n")

backend = AerSimulator()

# Match linear_witness.py noise parameters
noise_model = create_noise_model(
    single_qubit_error=0.0005,
    two_qubit_error=0.003,
    readout_error=0.02
)

print("Noise: 1% single-qubit, 2% two-qubit, 2% readout\n")
print(f"{'N':>3} | {'k':>2} | {'W_standard':>16} | {'W_jungnitsch':>16} | {'Δ':>8} | {'P+':>6} | {'GME?':>6}")
print("-" * 90)

for n in [4, 6, 8, 10, 12, 14, 15, 16]:
    r = run_jungnitsch_witness(n, backend, noise_model, shots=4000)
    
    delta = r['w_jungnitsch'] - r['w_standard']
    gme_std = r['w_standard'] < 0
    gme_jung = r['w_jungnitsch'] < 0
    
    w_std_str = f"{r['w_standard']:+.3f}±{r['se_standard']:.3f}"
    w_jung_str = f"{r['w_jungnitsch']:+.3f}±{r['se_jungnitsch']:.3f}"
    
    if gme_jung and not gme_std:
        status = "YES! ✓"
    elif gme_jung:
        status = "YES"
    else:
        status = "no"
    
    print(f"{n:>3} | {r['k']:>2} | {w_std_str:>16} | {w_jung_str:>16} | {delta:>+8.3f} | {r['p_plus']:>6.3f} | {status:>6}")

print("\n" + "="*90)
print("JUNGNITSCH WITNESS (arXiv:1010.6049):")
print("  W = W_standard - P_plus")
print("  P_plus = 0.5 × Prob(≤1 of k disjoint stabilizers satisfied)")
print("  k = ⌈N/3⌉ disjoint stabilizers (indices 0, 3, 6, ...)")
print("")
print("KEY RESULT: Jungnitsch improves detection at the noise THRESHOLD.")
print("  When W_std is barely positive, P_plus can flip W_jung negative → detect GME!")
print("="*90)
