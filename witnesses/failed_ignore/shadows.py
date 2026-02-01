"""
Shadow-Based Entanglement Depth Certification

Uses classical shadows to implement the ROBUST Tóth-Gühne Theorem 1 witness:
    W = (n-1)·I - Σ K_i
    
GME is certified if ⟨W⟩ < 0, equivalently: Σ⟨K_i⟩ > n-1

This is MUCH more noise-robust than Theorem 2 because:
- Stabilizers contribute ADDITIVELY (not multiplicatively)
- One weak stabilizer doesn't kill the entire certification
- We can certify entanglement depth over the best contiguous sub-chain

Shadow tomography advantage:
- Estimates all ⟨K_i⟩ from the SAME measurement data
- Only requires random Pauli measurements (no special settings needed)
- Can compute rigorous confidence intervals
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from scipy.stats import norm

# ============================
# 1. SETUP
# ============================

def create_noisy_backend(single_qubit_error=0.001, two_qubit_error=0.01, readout_error=0.02):
    """Creates noise model matching linear_witness.py for fair comparison."""
    noise_model = NoiseModel()
    
    err_1q = depolarizing_error(single_qubit_error, 1)
    noise_model.add_all_qubit_quantum_error(err_1q, ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 's', 'sdg'])
    
    err_2q = depolarizing_error(two_qubit_error, 2)
    noise_model.add_all_qubit_quantum_error(err_2q, ['cx', 'cz'])
    
    ro_err = ReadoutError([[1-readout_error, readout_error],
                           [readout_error, 1-readout_error]])
    noise_model.add_all_qubit_readout_error(ro_err)
    
    return AerSimulator(noise_model=noise_model)


def make_linear_cluster_state(n_qubits):
    """Create linear cluster state: H on all, CZ between neighbors."""
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for i in range(n_qubits - 1):
        qc.cz(i, i+1)
    return qc


# ============================
# 2. SHADOW TOMOGRAPHY
# ============================

def generate_pauli_shadows(qc_template, backend, n_shadows=10000, batch_size=1000):
    """
    Generate classical shadows using random Pauli measurements.
    
    Returns: List of (bases, outcomes) tuples
        bases[i] in {0,1,2} for {X,Y,Z} measurement on qubit i
        outcomes[i] in {0,1} for measurement result
    """
    n_qubits = qc_template.num_qubits
    shadows = []
    
    n_batches = (n_shadows + batch_size - 1) // batch_size
    
    # Pre-generate all random bases
    all_bases = [np.random.randint(0, 3, n_qubits) for _ in range(n_shadows)]
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_shadows)
        
        circuits = []
        bases_batch = all_bases[batch_start:batch_end]
        
        for bases in bases_batch:
            qc_shadow = qc_template.copy()
            for q in range(n_qubits):
                if bases[q] == 0:      # X basis
                    qc_shadow.h(q)
                elif bases[q] == 1:    # Y basis
                    qc_shadow.sdg(q)
                    qc_shadow.h(q)
                # Z basis: do nothing
            qc_shadow.measure_all()
            circuits.append(qc_shadow)
        
        # Transpile and run batch
        transpiled = transpile(circuits, backend, optimization_level=0)
        job = backend.run(transpiled, shots=1, memory=True)
        results = job.result()
        
        for i, bases in enumerate(bases_batch):
            bitstring = results.get_memory(i)[0]
            outcome_bits = [int(b) for b in bitstring[::-1]]
            shadows.append((bases, outcome_bits))
    
    return shadows


# ============================
# 3. STABILIZER ESTIMATION
# ============================

def estimate_stabilizer(shadow_data, target_pauli):
    """
    Estimate ⟨P⟩ for a Pauli string P using classical shadows.
    
    target_pauli: list of (qubit_index, pauli_type) where pauli_type in {'X','Y','Z'}
    
    For Pauli shadows, the unbiased estimator for a k-local Pauli is:
        ô(P) = 3^k × ∏_{i} (eigenvalue if basis matches, else 0)
    
    IMPORTANT: We must include ALL samples (zeros for non-matching bases)
    to get an unbiased estimate. The 3^k factor compensates for the 
    (1/3)^k probability of matching.
    
    Returns: list of individual estimates (for statistical analysis)
    """
    map_p = {'X': 0, 'Y': 1, 'Z': 2}
    k = len(target_pauli)
    
    estimates = []
    for bases, outcomes in shadow_data:
        # Check if all bases match
        match = True
        val = 1
        for q_idx, p_type in target_pauli:
            if bases[q_idx] != map_p[p_type]:
                match = False
                break
            eigenval = 1 if outcomes[q_idx] == 0 else -1
            val *= eigenval
        
        if match:
            # Matching sample: contributes 3^k × eigenvalue
            estimates.append((3 ** k) * val)
        else:
            # Non-matching sample: contributes 0
            estimates.append(0.0)
    
    return estimates


def get_linear_cluster_stabilizers(n_qubits):
    """
    Get stabilizer generators for n-qubit linear cluster state.
    
    K_0 = X_0 Z_1
    K_i = Z_{i-1} X_i Z_{i+1}  for 1 ≤ i ≤ n-2
    K_{n-1} = Z_{n-2} X_{n-1}
    """
    stabilizers = []
    
    # First: X_0 Z_1
    stabilizers.append([(0, 'X'), (1, 'Z')])
    
    # Middle: Z_{i-1} X_i Z_{i+1}
    for i in range(1, n_qubits - 1):
        stabilizers.append([(i-1, 'Z'), (i, 'X'), (i+1, 'Z')])
    
    # Last: Z_{n-2} X_{n-1}
    stabilizers.append([(n_qubits-2, 'Z'), (n_qubits-1, 'X')])
    
    return stabilizers


# ============================
# 4. THEOREM 1 WITNESS (ROBUST)
# ============================

def theorem1_witness_shadows(n_qubits, shadow_data, confidence=0.95):
    """
    Implement Tóth-Gühne Theorem 1 witness using shadows.
    
    Witness: W = (n-1)·I - Σ_{i=0}^{n-1} K_i
    
    GME certified if ⟨W⟩ < 0, i.e., Σ⟨K_i⟩ > n-1
    
    Returns dict with:
        - stab_estimates: individual ⟨K_i⟩ estimates
        - sum_estimate: Σ⟨K_i⟩
        - sum_ci_lower: lower confidence bound on sum
        - witness: W = (n-1) - Σ⟨K_i⟩
        - witness_ci_upper: upper confidence bound on W
        - certified: True if W upper bound < 0
    """
    stabilizers = get_linear_cluster_stabilizers(n_qubits)
    
    # Estimate each stabilizer
    all_stab_estimates = []
    stab_means = []
    stab_stderrs = []
    
    for stab in stabilizers:
        estimates = estimate_stabilizer(shadow_data, stab)
        if len(estimates) > 0:
            mean = np.mean(estimates)
            stderr = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
        else:
            mean = 0.0
            stderr = 1.0  # Conservative
        
        all_stab_estimates.append(estimates)
        stab_means.append(mean)
        stab_stderrs.append(stderr)
    
    # Sum of stabilizer expectations
    sum_estimate = sum(stab_means)
    
    # Standard error of sum (assuming independence between estimates)
    # This is conservative - actual correlation might reduce error
    sum_stderr = np.sqrt(sum(se**2 for se in stab_stderrs))
    
    # Confidence interval
    z = norm.ppf(0.5 + confidence / 2)
    sum_ci_lower = sum_estimate - z * sum_stderr
    sum_ci_upper = sum_estimate + z * sum_stderr
    
    # Witness value: W = (n-1) - Σ⟨K_i⟩
    witness = (n_qubits - 1) - sum_estimate
    witness_ci_upper = (n_qubits - 1) - sum_ci_lower
    
    return {
        'stab_estimates': stab_means,
        'stab_stderrs': stab_stderrs,
        'sum_estimate': sum_estimate,
        'sum_stderr': sum_stderr,
        'sum_ci_lower': sum_ci_lower,
        'sum_ci_upper': sum_ci_upper,
        'witness': witness,
        'witness_ci_upper': witness_ci_upper,
        'certified': witness_ci_upper < 0,
        'threshold': n_qubits - 1,
        'confidence': confidence
    }


# ============================
# 5. ENTANGLEMENT DEPTH (SUB-CHAIN)
# ============================

def find_best_subchain_theorem1(stab_means, stab_stderrs, confidence=0.95):
    """
    Find the longest contiguous sub-chain certifiable via Theorem 1.
    
    For sub-chain [a, b] with m = b-a+1 stabilizers:
    Certify if Σ_{i=a}^{b} ⟨K_i⟩ > m - 1 (with confidence)
    
    Returns: (best_length, best_start, best_end, best_witness_upper)
    """
    n = len(stab_means)
    z = norm.ppf(0.5 + confidence / 2)
    
    best_length = 0
    best_start = 0
    best_end = 0
    best_witness_upper = float('inf')
    
    for start in range(n):
        for end in range(start + 1, n + 1):  # end is exclusive
            m = end - start  # number of stabilizers in chain
            
            # Sum and stderr for this sub-chain
            sub_sum = sum(stab_means[start:end])
            sub_stderr = np.sqrt(sum(se**2 for se in stab_stderrs[start:end]))
            
            # Lower CI bound on sum
            sub_ci_lower = sub_sum - z * sub_stderr
            
            # Witness: W = (m-1) - Σ⟨K_i⟩
            # Upper bound: W_upper = (m-1) - sub_ci_lower
            witness_upper = (m - 1) - sub_ci_lower
            
            # Certified if witness upper bound < 0
            if witness_upper < 0:
                if m > best_length:
                    best_length = m
                    best_start = start
                    best_end = end - 1  # inclusive
                    best_witness_upper = witness_upper
    
    return best_length, best_start, best_end, best_witness_upper


# ============================
# 6. MAIN
# ============================

if __name__ == "__main__":
    print("=" * 60)
    print("ROBUST Entanglement Witness via Shadow Tomography")
    print("Using Tóth-Gühne Theorem 1: W = (n-1)·I - Σ K_i")
    print("=" * 60)
    
    # Noise parameters matching linear_witness.py
    print("\nNoise Parameters:")
    print("  Single-qubit gates: 0.1% error")
    print("  Two-qubit gates: 1.0% error")
    print("  Readout: 2.0% error")
    
    backend = create_noisy_backend(
        single_qubit_error=0.001,
        two_qubit_error=0.01,
        readout_error=0.02
    )
    
    N_SHADOWS = 5000  # Reduced for speed; increase for better accuracy
    CONFIDENCE = 0.95
    
    print(f"\nUsing {N_SHADOWS} shadows, {CONFIDENCE*100:.0f}% confidence\n")
    print("-" * 60)
    
    results_summary = []
    
    for n_qubits in [8, 12, 16, 20, 30]:  # Reduced list for faster testing
        print(f"\n=== N = {n_qubits} qubits ===")
        
        # Generate shadows
        qc = make_linear_cluster_state(n_qubits)
        print(f"Generating shadows...", end=" ", flush=True)
        shadows = generate_pauli_shadows(qc, backend, n_shadows=N_SHADOWS)
        print("done")
        
        # Full-state Theorem 1 witness
        result = theorem1_witness_shadows(n_qubits, shadows, confidence=CONFIDENCE)
        
        print(f"\nFull State (Theorem 1):")
        print(f"  Σ⟨K_i⟩ = {result['sum_estimate']:.3f} ± {result['sum_stderr']:.3f}")
        print(f"  Threshold: {result['threshold']}")
        print(f"  95% CI: [{result['sum_ci_lower']:.3f}, {result['sum_ci_upper']:.3f}]")
        print(f"  Witness W = {result['witness']:.3f}")
        print(f"  W upper bound = {result['witness_ci_upper']:.3f}")
        
        if result['certified']:
            print(f"  ✓ FULL GME CERTIFIED ({n_qubits} qubits)")
        else:
            print(f"  ✗ Full GME not certified")
        
        # Find best sub-chain
        depth, start, end, w_upper = find_best_subchain_theorem1(
            result['stab_estimates'], 
            result['stab_stderrs'],
            confidence=CONFIDENCE
        )
        
        print(f"\nEntanglement Depth (best sub-chain):")
        if depth > 0:
            print(f"  Certified depth: {depth} qubits")
            print(f"  Sub-chain: qubits {start} to {end}")
            print(f"  W upper bound: {w_upper:.3f}")
        else:
            print(f"  No sub-chain certifiable at {CONFIDENCE*100:.0f}% confidence")
        
        results_summary.append({
            'n': n_qubits,
            'full_certified': result['certified'],
            'depth': depth,
            'window': (start, end) if depth > 0 else None
        })
        
        # Show individual stabilizer values
        print(f"\nIndividual ⟨K_i⟩ (first 5, last 5):")
        ests = result['stab_estimates']
        for i in range(min(5, len(ests))):
            print(f"  K_{i}: {ests[i]:.3f}")
        if len(ests) > 10:
            print(f"  ...")
        for i in range(max(5, len(ests)-5), len(ests)):
            print(f"  K_{i}: {ests[i]:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'N':>4} | {'Full GME':>10} | {'Depth':>6} | {'Window':>12}")
    print("-" * 40)
    for r in results_summary:
        full = "✓ YES" if r['full_certified'] else "✗ NO"
        window = f"[{r['window'][0]},{r['window'][1]}]" if r['window'] else "-"
        print(f"{r['n']:>4} | {full:>10} | {r['depth']:>6} | {window:>12}")
    
    print("\n" + "=" * 60)
    print("WHY THEOREM 1 IS MORE ROBUST THAN THEOREM 2:")
    print("=" * 60)
    print("""
Theorem 2 (linear_witness.py):
  W = 3 - 2(P_A + P_B)
  where P_A = Prob(ALL even stabilizers = +1)
  
  Problem: If ANY single stabilizer fails, the entire product
  collapses. This is exponentially sensitive to noise.

Theorem 1 (this implementation):
  W = (n-1) - Σ⟨K_i⟩
  
  Advantage: Stabilizers contribute ADDITIVELY. A few weak
  stabilizers just reduce the sum slightly, they don't kill
  the entire certification.
  
  Shadow tomography bonus: We estimate all ⟨K_i⟩ from the
  SAME random measurements, no special settings needed.
""")
