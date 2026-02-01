import numpy as np
from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
# from qrisp.interface import IQMBackend # Uncomment for real hardware

def run_cluster_witness_theorem_2(n_qubits, backend, shots=2000):
    """
    Implements the Cluster State Witness from Toth & Guhne Theorem 2.
    Formula: W = 3*I - 2*(Prob_All_Even_Satisfied + Prob_All_Odd_Satisfied)
    Target: W < 0 implies Genuine Multipartite Entanglement.
    """
    
    # --- 1. PREPARE THE STATE ---
    # We create the "perfect" linear cluster state to test against
    qc_base = QuantumCircuit(n_qubits)
    qc_base.h(range(n_qubits)) # Initialize |+>
    # Apply CZ gates (Linear Topology matches IQM Garnet)
    for i in range(0, n_qubits - 1, 2): qc_base.cz(i, i+1) # Even edges
    for i in range(1, n_qubits - 1, 2): qc_base.cz(i, i+1) # Odd edges

    # --- 2. DEFINE THE TWO SETTINGS (Fig 1c from Paper) ---
    
    # Setting A: Measure stabilizers centered on EVEN QISKIT INDICES (0, 2, 4...)
    # Corresponds to Paper's "Odd k" (S1, S3...)
    # Pattern: X Z X Z ...
    qc_A = qc_base.copy()
    # To measure X, apply H. To measure Z, do nothing.
    for i in range(0, n_qubits, 2): 
        qc_A.h(i) 
    qc_A.measure_all()
    
    # Setting B: Measure stabilizers centered on ODD QISKIT INDICES (1, 3, 5...)
    # Corresponds to Paper's "Even k" (S2, S4...)
    # Pattern: Z X Z X ...
    qc_B = qc_base.copy()
    for i in range(1, n_qubits, 2): 
        qc_B.h(i)
    qc_B.measure_all()
    
    # --- 3. RUN EXPERIMENT ---
    print(f"Running Witness for N={n_qubits}...")
    # fig_A = qc_A.draw(output="mpl")
    # plt.show()
    job_A = backend.run(transpile(qc_A, backend), shots=shots)
    # fig_B = qc_B.draw(output="mpl")
    # plt.show()
    job_B = backend.run(transpile(qc_B, backend), shots=shots)
    counts_A = job_A.result().get_counts()
    counts_B = job_B.result().get_counts()
    
    # --- 4. CALCULATE PROBABILITIES (The "Product" Terms) ---
    
    # Term 1: Probability that ALL "Odd k" stabilizers are +1
    # Stabilizers: S_0, S_2, S_4 (Qiskit indices)
    success_count_A = 0
    for bitstring, count in counts_A.items():
        all_satisfied = True
        # Check every stabilizer in this group
        for i in range(0, n_qubits, 2): 
            # Calculate Parity of S_i = Z_{i-1} X_i Z_{i+1}
            # Note: Qiskit bitstrings are reversed! bit[0] is qubit N-1.
            parity = 1
            # Center (X_i)
            if bitstring[n_qubits - 1 - i] == '1': parity *= -1
            # Left (Z_{i-1})
            if i > 0 and bitstring[n_qubits - 1 - (i-1)] == '1': parity *= -1
            # Right (Z_{i+1})
            if i < n_qubits - 1 and bitstring[n_qubits - 1 - (i+1)] == '1': parity *= -1
            
            if parity == -1: # Rule Broken!
                all_satisfied = False
                break
        
        if all_satisfied:
            success_count_A += count
            
    prob_A = success_count_A / shots

    # Term 2: Probability that ALL "Even k" stabilizers are +1
    # Stabilizers: S_1, S_3, S_5 (Qiskit indices)
    success_count_B = 0
    for bitstring, count in counts_B.items():
        all_satisfied = True
        for i in range(1, n_qubits, 2): 
            parity = 1
            # Center (X_i)
            if bitstring[n_qubits - 1 - i] == '1': parity *= -1
            # Left (Z_{i-1})
            if bitstring[n_qubits - 1 - (i-1)] == '1': parity *= -1
            # Right (Z_{i+1})
            if i < n_qubits - 1 and bitstring[n_qubits - 1 - (i+1)] == '1': parity *= -1
            
            if parity == -1: 
                all_satisfied = False
                break
                
        if all_satisfied:
            success_count_B += count
            
    prob_B = success_count_B / shots

    # --- 5. COMPUTE WITNESS VALUE ---
    # W = 3 - 2 * (P_A + P_B)
    w_value = 3 - 2 * (prob_A + prob_B)
    
    return w_value, prob_A, prob_B

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

# --- MOCK EXECUTION ---
# Create a noisy simulator
print("=== Noisy Simulation ===")
print("Noise Parameters:")
print("  Single-qubit gates (H): 0.1% error")
print("  Two-qubit gates (CZ): 1.0% error")
print("  Readout: 2.0% error\n")

noise_model = create_noise_model(
     single_qubit_error=0.01,
        two_qubit_error=0.02,
        readout_error=0.02
)
backend_sim = AerSimulator(noise_model=noise_model)

# Test for N=4, 6, 8, 10
for n in [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 20, 25, 30]:
    w, pA, pB = run_cluster_witness_theorem_2(n, backend_sim)
    print(f"N={n}: Witness = {w:.3f} (P_odd={pA:.2f}, P_even={pB:.2f})")
    # Expected for perfect state: W = 3 - 2(1+1) = -1.0
    # With noise, W will be closer to 0 (less negative)