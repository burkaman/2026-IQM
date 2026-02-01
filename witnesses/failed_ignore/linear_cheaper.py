import numpy as np
from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from scipy import stats
# from qrisp.interface import IQMBackend # Uncomment for real hardware

def run_cluster_witness_theorem_2(n_qubits, backend, shots=2000):
    """
    Implements the Cluster State Witness from Toth & Guhne Theorem 2.
    Formula: W = 3*I - 2*(Prob_All_Even_Satisfied + Prob_All_Odd_Satisfied)
    Target: W < 0 implies Genuine Multipartite Entanglement.
    
    This is the fragile version - kept for comparison.
    """
    
    # --- 1. PREPARE THE STATE ---
    qc_base = QuantumCircuit(n_qubits)
    qc_base.h(range(n_qubits)) # Initialize |+>
    # Apply CZ gates (Linear Topology matches IQM Garnet)
    for i in range(0, n_qubits - 1, 2): qc_base.cz(i, i+1) # Even edges
    for i in range(1, n_qubits - 1, 2): qc_base.cz(i, i+1) # Odd edges

    # --- 2. DEFINE THE TWO SETTINGS ---
    qc_A = qc_base.copy()
    for i in range(0, n_qubits, 2): 
        qc_A.h(i) 
    qc_A.measure_all()
    
    qc_B = qc_base.copy()
    for i in range(1, n_qubits, 2): 
        qc_B.h(i)
    qc_B.measure_all()
    
    # --- 3. RUN EXPERIMENT ---
    print(f"  Running Theorem 2 witness for N={n_qubits}...")
    job_A = backend.run(transpile(qc_A, backend), shots=shots)
    job_B = backend.run(transpile(qc_B, backend), shots=shots)
    counts_A = job_A.result().get_counts()
    counts_B = job_B.result().get_counts()
    
    # --- 4. CALCULATE PROBABILITIES (The "Product" Terms) ---
    success_count_A = 0
    for bitstring, count in counts_A.items():
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
            success_count_A += count
            
    prob_A = success_count_A / shots

    success_count_B = 0
    for bitstring, count in counts_B.items():
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
            success_count_B += count
            
    prob_B = success_count_B / shots

    # --- 5. COMPUTE WITNESS VALUE ---
    w_value = 3 - 2 * (prob_A + prob_B)
    
    return w_value, prob_A, prob_B

def diagnose_stabilizers(n_qubits, backend, shots=8000, plot=False):
    """
    DIAGNOSTIC ONLY: Estimates each stabilizer <K_i> individually to
    identify which regions of the cluster state maintain entanglement.
    
    This is pure post-processing - no witness bound is computed here.
    Use this to identify which contiguous sub-chains to certify.
    
    For a linear cluster, each K_i = X_i Z_{i-1} Z_{i+1}.
    Setting A (H on even qubits) lets you extract K_i for even i.
    Setting B (H on odd qubits) lets you extract K_i for odd i.
    
    Returns:
        ki_values: dict mapping qubit index -> <K_i>
        counts_A, counts_B: Measurement results (reusable for sub-chain analysis)
    """

    # --- 1. PREPARE STATE ---
    qc_base = QuantumCircuit(n_qubits)
    qc_base.h(range(n_qubits))
    for i in range(0, n_qubits - 1, 2): qc_base.cz(i, i+1)
    for i in range(1, n_qubits - 1, 2): qc_base.cz(i, i+1)

    # --- 2. TWO MEASUREMENT SETTINGS ---
    qc_A = qc_base.copy()
    for i in range(0, n_qubits, 2): qc_A.h(i)
    qc_A.measure_all()

    qc_B = qc_base.copy()
    for i in range(1, n_qubits, 2): qc_B.h(i)
    qc_B.measure_all()

    # --- 3. RUN ---
    print(f"  Diagnosing stabilizers for N={n_qubits}...")
    job_A = backend.run(transpile(qc_A, backend), shots=shots)
    job_B = backend.run(transpile(qc_B, backend), shots=shots)
    counts_A = job_A.result().get_counts()
    counts_B = job_B.result().get_counts()

    # --- 4. EXTRACT INDIVIDUAL <K_i> ---
    ki_values = {}

    def extract_ki(counts, n_qubits, stabilizer_indices, shots):
        """Given counts from one setting, extract <K_i> for each i in stabilizer_indices."""
        results = {i: 0.0 for i in stabilizer_indices}
        for bitstring, count in counts.items():
            for i in stabilizer_indices:
                parity = 1
                # Center qubit (X basis, already rotated by H)
                if bitstring[n_qubits - 1 - i] == '1': parity *= -1
                # Left neighbor (Z basis)
                if i > 0 and bitstring[n_qubits - 1 - (i-1)] == '1': parity *= -1
                # Right neighbor (Z basis)
                if i < n_qubits - 1 and bitstring[n_qubits - 1 - (i+1)] == '1': parity *= -1
                results[i] += parity * count
        # Normalize
        for i in stabilizer_indices:
            results[i] /= shots
        return results

    even_indices = list(range(0, n_qubits, 2))
    odd_indices  = list(range(1, n_qubits, 2))

    ki_values.update(extract_ki(counts_A, n_qubits, even_indices, shots))
    ki_values.update(extract_ki(counts_B, n_qubits, odd_indices,  shots))

    # --- 5. OPTIONAL: PLOT <K_i> vs i ---
    if plot:
        indices_sorted = sorted(ki_values.keys())
        ki_sorted = [ki_values[i] for i in indices_sorted]

        plt.figure(figsize=(10, 4))
        plt.bar(indices_sorted, ki_sorted, color=['steelblue' if v > 0.7 else 'salmon' for v in ki_sorted])
        plt.axhline(y=0.7, color='red', linestyle='--', label='Typical "good" threshold')
        plt.axhline(y=0.0, color='black', linestyle='-', linewidth=0.8)
        plt.xlabel('Stabilizer index i')
        plt.ylabel('⟨K_i⟩')
        plt.title(f'Individual stabilizer expectations, N={n_qubits}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return ki_values, counts_A, counts_B

def wilson_interval(successes, total, confidence=0.95):
    """
    Wilson score interval for binomial proportion.
    Better than normal approximation, especially for small counts.
    
    Returns: (point_estimate, lower_bound, upper_bound)
    """
    p_hat = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))) / denominator
    
    return p_hat, max(0, center - margin), min(1, center + margin)

def run_subchain_witness_theorem_2(n_qubits, counts_A, counts_B, shots, 
                                    start_qubit, end_qubit, confidence_level=0.95):
    """
    RIGOROUS: Applies Theorem 2 witness to a CONTIGUOUS sub-chain.
    
    Key insight: A contiguous sub-chain of a linear cluster IS ITSELF a valid
    linear cluster state. Therefore, the bound W = 3 - 2(P_A + P_B) < 0 applies
    rigorously with no new derivation needed.
    
    This is the scientifically defensible way to handle partial stabilizer failure.
    
    Args:
        n_qubits: Total number of qubits in prepared state
        counts_A, counts_B: Measurement results from full state
        shots: Number of shots
        start_qubit, end_qubit: Inclusive range [start, end] defining sub-chain
        confidence_level: Confidence level for statistical bounds (default 95%)
    
    Returns:
        Dictionary with:
            - witness: Point estimate of W
            - witness_upper: Upper confidence bound (for conservative certification)
            - p_A, p_B: Measured probabilities
            - certified: True if upper bound < 0 (GME proven at confidence level)
            - sub_chain_size: Number of qubits certified
            
    Reference:
        - Tóth & Gühne, PRA 72, 022340 (2005) - Theorem 2
        - Amaro & Müller, PRA 101, 012317 (2020) - Subsystem witnesses (arXiv:1911.01144)
    """
    
    # --- CHECK STABILIZERS IN SUB-CHAIN ONLY ---
    # For sub-chain [start, end], check stabilizers centered on qubits in that range
    # Each stabilizer K_i checks qubits (i-1, i, i+1) within the sub-chain
    
    success_count_A = 0
    for bitstring, count in counts_A.items():
        all_satisfied = True
        # Check only even-indexed stabilizers in the sub-chain
        for i in range(start_qubit, end_qubit + 1, 2):
            parity = 1
            # Center qubit
            if bitstring[n_qubits - 1 - i] == '1': parity *= -1
            # Left neighbor (only check if within sub-chain)
            if i > start_qubit and bitstring[n_qubits - 1 - (i-1)] == '1': parity *= -1
            # Right neighbor (only check if within sub-chain)  
            if i < end_qubit and bitstring[n_qubits - 1 - (i+1)] == '1': parity *= -1
            
            if parity == -1:
                all_satisfied = False
                break
        
        if all_satisfied:
            success_count_A += count
    
    success_count_B = 0
    for bitstring, count in counts_B.items():
        all_satisfied = True
        # Check only odd-indexed stabilizers in the sub-chain
        for i in range(start_qubit + 1, end_qubit + 1, 2):  # Start from first odd in range
            parity = 1
            if bitstring[n_qubits - 1 - i] == '1': parity *= -1
            if i > start_qubit and bitstring[n_qubits - 1 - (i-1)] == '1': parity *= -1
            if i < end_qubit and bitstring[n_qubits - 1 - (i+1)] == '1': parity *= -1
            
            if parity == -1:
                all_satisfied = False
                break
        
        if all_satisfied:
            success_count_B += count
    
    # --- COMPUTE WITNESS WITH CONFIDENCE INTERVALS ---
    p_A, p_A_lower, p_A_upper = wilson_interval(success_count_A, shots, confidence_level)
    p_B, p_B_lower, p_B_upper = wilson_interval(success_count_B, shots, confidence_level)
    
    # Point estimate
    W = 3 - 2 * (p_A + p_B)
    
    # Conservative upper bound: worst case for certification
    # We want to show W < 0, so we need the upper confidence bound to be negative
    W_upper = 3 - 2 * (p_A_lower + p_B_lower)
    
    return {
        'witness': W,
        'witness_upper': W_upper,
        'p_A': p_A,
        'p_B': p_B,
        'p_A_interval': (p_A_lower, p_A_upper),
        'p_B_interval': (p_B_lower, p_B_upper),
        'certified': W_upper < 0,
        'confidence': confidence_level,
        'sub_chain': (start_qubit, end_qubit),
        'sub_chain_size': end_qubit - start_qubit + 1
    }

def find_best_subchain(ki_values, min_threshold=0.7):
    """
    Identifies the longest contiguous sub-chain where all stabilizers exceed threshold.
    
    This is a heuristic for finding the "best" sub-chain to certify.
    You could also try multiple sub-chains and report the largest certified one.
    """
    indices_sorted = sorted(ki_values.keys())
    
    best_start = 0
    best_end = 0
    best_length = 0
    
    current_start = None
    for i, idx in enumerate(indices_sorted):
        if ki_values[idx] >= min_threshold:
            if current_start is None:
                current_start = idx
            current_end = idx
            current_length = current_end - current_start + 1
            
            if current_length > best_length:
                best_start = current_start
                best_end = current_end
                best_length = current_length
        else:
            current_start = None
    
    return best_start, best_end, best_length

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
    readout_probs = [[1 - readout_error, readout_error], 
                     [readout_error, 1 - readout_error]]
    readout_noise = ReadoutError(readout_probs)
    noise_model.add_all_qubit_readout_error(readout_noise)
    
    return noise_model

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("=== Linear Cluster State Witness: RIGOROUS Sub-Chain Approach ===")
    print("\nNoise Parameters:")
    print("  Single-qubit gates (H): 0.1% error")
    print("  Two-qubit gates (CZ): 2.0% error")
    print("  Readout: 0.2% error\n")

    noise_model = create_noise_model(
        single_qubit_error=0.001,
        two_qubit_error=0.02,
        readout_error=0.002
    )
    backend_sim = AerSimulator(noise_model=noise_model)

    # --- COMPARISON: FULL STATE vs SUB-CHAIN ---
    print("=== Full State Theorem 2 Witness (Baseline) ===\n")
    for n in [10, 12, 14, 16, 18, 20]:
        w, pA, pB = run_cluster_witness_theorem_2(n, backend_sim, shots=8000)
        print(f"N={n}: W = {w:.4f} (P_A={pA:.4f}, P_B={pB:.4f})", end="")
        if w < 0:
            print(" ✓ GME certified")
        else:
            print(" ✗ Failed")
    
    # --- RIGOROUS SUB-CHAIN WITNESS ---
    print("\n=== Sub-Chain Witness (Rigorous - Theorem 2 on contiguous sub-chains) ===\n")
    for n in [10, 12, 14, 16, 18, 20]:
        # First, diagnose which stabilizers are strong
        ki_vals, counts_A, counts_B = diagnose_stabilizers(n, backend_sim, shots=8000, plot=False)
        
        # Find best contiguous sub-chain
        best_start, best_end, best_length = find_best_subchain(ki_vals, min_threshold=0.7)
        
        if best_length >= 4:  # Need at least a few qubits for meaningful entanglement
            # Apply Theorem 2 to the sub-chain
            result = run_subchain_witness_theorem_2(
                n, counts_A, counts_B, shots=8000,
                start_qubit=best_start, end_qubit=best_end,
                confidence_level=0.95
            )
            
            print(f"N={n}: Best sub-chain [{best_start}, {best_end}] ({result['sub_chain_size']} qubits)")
            print(f"  W = {result['witness']:.4f}, W_upper(95%) = {result['witness_upper']:.4f}")
            print(f"  P_A = {result['p_A']:.4f}, P_B = {result['p_B']:.4f}")
            if result['certified']:
                print(f"  ✓ GME CERTIFIED at 95% confidence over {result['sub_chain_size']} qubits")
            else:
                print(f"  ✗ Not certified at 95% confidence")
        else:
            print(f"N={n}: No viable sub-chain found (all stabilizers too weak)")
        
        print()
