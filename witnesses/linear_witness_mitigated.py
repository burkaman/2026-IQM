"""
Linear Cluster State Witness with Error Mitigation

Mitigation techniques (each mathematically independent):
1. Readout Error Mitigation - calibrates and corrects measurement errors (LINEAR)
2. Zero Noise Extrapolation (ZNE) - runs at multiple noise levels, extrapolates to zero

Reference: Toth & Guhne Theorem 2
Witness: W = 3 - 2*(P_A + P_B) where P_A, P_B are probabilities all stabilizers satisfied

For real hardware: Set use_simulator=False and provide a real backend
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from typing import Dict, List, Tuple, Optional

# Mitiq imports
from mitiq import zne

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def bit_at_qubit(bitstring: str, q: int, n: int) -> int:
    """Extract bit for qubit q from Qiskit bitstring (little-endian: rightmost = q0)."""
    return 1 if bitstring[n - 1 - q] == "1" else 0

def stabilizer_eigenvalue(bitstring: str, i: int, n: int) -> int:
    """
    Compute eigenvalue of cluster state stabilizer K_i = Z_{i-1} X_i Z_{i+1}.
    
    For the measurement setting where qubit i is measured in X basis (after H),
    and neighbors in Z basis, the eigenvalue is:
        +1 if parity(bit_i ⊕ bit_{i-1} ⊕ bit_{i+1}) = 0
        -1 if parity = 1
    
    Returns +1 or -1.
    """
    parity = bit_at_qubit(bitstring, i, n)  # X measurement result
    if i > 0:
        parity ^= bit_at_qubit(bitstring, i - 1, n)  # Z neighbor
    if i < n - 1:
        parity ^= bit_at_qubit(bitstring, i + 1, n)  # Z neighbor
    return +1 if parity == 0 else -1

# ============================================================
# TECHNIQUE 1: READOUT ERROR MITIGATION (LINEAR)
# ============================================================
#
# HOW IT WORKS:
# 1. Calibration: Prepare |0⟩^n and |1⟩^n, measure to get error rates
# 2. Build confusion matrix: M where M[i,j] = P(measure i | prepared j)
# 3. Correction: Apply M^(-1) to measured probabilities
#
# For simplicity, we use a symmetric model:
#   p(0|1) = p(1|0) = ε (average error rate)
#   Correction: p_true = (p_measured - ε) / (1 - 2ε)
#
# VALIDITY: Standard technique. Works well when errors are relatively uniform.
# For real hardware, consider per-qubit calibration.
# ============================================================

def calibrate_readout(backend, n_qubits: int, shots: int = 2000, 
                      noise_model=None) -> Tuple[float, float]:
    """
    Calibrate readout errors by preparing and measuring |0...0⟩ and |1...1⟩.
    
    Returns:
        (p_correct_0, p_correct_1): Probabilities of correctly measuring each state
    """
    # Prepare |0...0⟩
    qc0 = QuantumCircuit(n_qubits)
    qc0.measure_all()
    
    # Prepare |1...1⟩  
    qc1 = QuantumCircuit(n_qubits)
    qc1.x(range(n_qubits))
    qc1.measure_all()
    
    # Run calibration circuits
    if noise_model:
        result0 = backend.run(transpile(qc0, backend), shots=shots, noise_model=noise_model).result()
        result1 = backend.run(transpile(qc1, backend), shots=shots, noise_model=noise_model).result()
    else:
        result0 = backend.run(transpile(qc0, backend), shots=shots).result()
        result1 = backend.run(transpile(qc1, backend), shots=shots).result()
    
    all_zeros = '0' * n_qubits
    all_ones = '1' * n_qubits
    
    p_correct_0 = result0.get_counts().get(all_zeros, 0) / shots
    p_correct_1 = result1.get_counts().get(all_ones, 0) / shots
    
    return p_correct_0, p_correct_1

def apply_readout_correction(prob: float, p0: float, p1: float) -> float:
    """
    Apply LINEAR readout error correction using symmetric error model.
    
    Model: p_measured = (1-ε)*p_true + ε*(1-p_true) where ε = average error rate
    Inversion: p_true = (p_measured - ε) / (1 - 2ε)
    
    This is a LINEAR transformation of the measured probability.
    """
    # Average error rate
    epsilon = 1 - (p0 + p1) / 2
    
    if epsilon >= 0.5:
        # Error rate too high, correction would be unstable
        return prob
    
    corrected = (prob - epsilon) / (1 - 2 * epsilon)
    return max(0.0, min(1.0, corrected))  # Clamp to valid probability

# ============================================================
# TECHNIQUE 2: ZERO NOISE EXTRAPOLATION (ZNE) via mitiq
# ============================================================
#
# HOW IT WORKS:
# 1. Run circuit at base noise level → get result r1
# 2. Artificially increase noise (fold gates) → get results r2, r3, ...
# 3. Fit polynomial to (noise_scale, result) pairs
# 4. Extrapolate to noise_scale = 0
#
# Gate folding: Replace G with G·G†·G (triples the noise)
# fold_global folds entire circuit uniformly
#
# VALIDITY: Well-established technique. Works best when:
# - Noise is relatively small (so extrapolation is stable)
# - Noise scales predictably with gate count
# ============================================================

def make_executor_for_zne(backend, shots: int, n_qubits: int, setting: str, noise_model=None):
    """
    Create executor function for mitiq ZNE.
    Returns probability that all stabilizers in the setting are satisfied.
    """
    def executor(circuit: QuantumCircuit) -> float:
        # Add measurements if needed
        if circuit.num_clbits == 0:
            circuit = circuit.copy()
            circuit.measure_all()
        
        # Execute
        if noise_model:
            result = backend.run(transpile(circuit, backend), shots=shots, 
                                noise_model=noise_model).result()
        else:
            result = backend.run(transpile(circuit, backend), shots=shots).result()
        
        counts = result.get_counts()
        
        # Compute probability all target stabilizers satisfied
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        # Setting A: even indices (0, 2, 4, ...)
        # Setting B: odd indices (1, 3, 5, ...)
        target_indices = range(0, n_qubits, 2) if setting == 'A' else range(1, n_qubits, 2)
        
        success = 0
        for bitstring, count in counts.items():
            if all(stabilizer_eigenvalue(bitstring, i, n_qubits) == +1 for i in target_indices):
                success += count
        
        return success / total
    
    return executor

# ============================================================
# CIRCUIT BUILDERS
# ============================================================

def make_cluster_state_circuit(n: int) -> QuantumCircuit:
    """
    Create linear cluster state: |C_n⟩ = CZ_{01}·CZ_{12}·...·CZ_{n-2,n-1} · H^⊗n |0⟩^n
    
    The cluster state satisfies stabilizers K_i = Z_{i-1} X_i Z_{i+1} for all i.
    """
    qc = QuantumCircuit(n)
    qc.h(range(n))  # Initialize all in |+⟩
    # Apply CZ gates (can be done in 2 layers for linear topology)
    for i in range(0, n - 1, 2): qc.cz(i, i + 1)  # Even edges
    for i in range(1, n - 1, 2): qc.cz(i, i + 1)  # Odd edges
    return qc

def make_measurement_circuit_A(n: int) -> QuantumCircuit:
    """
    Circuit to measure even-indexed stabilizers (K_0, K_2, K_4, ...).
    Apply H to even qubits (to measure X), leave odd qubits (to measure Z).
    """
    qc = make_cluster_state_circuit(n)
    for i in range(0, n, 2):
        qc.h(i)
    qc.measure_all()
    return qc

def make_measurement_circuit_B(n: int) -> QuantumCircuit:
    """
    Circuit to measure odd-indexed stabilizers (K_1, K_3, K_5, ...).
    Apply H to odd qubits (to measure X), leave even qubits (to measure Z).
    """
    qc = make_cluster_state_circuit(n)
    for i in range(1, n, 2):
        qc.h(i)
    qc.measure_all()
    return qc

# ============================================================
# MAIN WITNESS FUNCTION
# ============================================================

def run_witness(
    n_qubits: int,
    backend,
    shots: int = 2000,
    noise_model = None,  # None for real hardware
    use_readout_mitigation: bool = False,
    use_zne: bool = False,
    zne_scale_factors: List[float] = [1.0, 2.0, 3.0],
    verbose: bool = False,
) -> Dict:
    """
    Run the Toth-Guhne cluster state witness with optional error mitigation.
    
    Witness formula: W = 3 - 2*(P_A + P_B)
    where P_A = Prob(all even stabilizers satisfied)
          P_B = Prob(all odd stabilizers satisfied)
    
    W < 0 implies Genuine Multipartite Entanglement (GME).
    
    Args:
        n_qubits: Number of qubits
        backend: Qiskit backend (simulator or real hardware)
        shots: Number of measurement shots
        noise_model: Noise model for simulator (None for real hardware)
        use_readout_mitigation: Enable LINEAR readout error correction
        use_zne: Enable Zero Noise Extrapolation via mitiq
        zne_scale_factors: Noise scaling factors for ZNE
        verbose: Print detailed progress
    
    Returns:
        Dictionary with witness value, probabilities, and diagnostics
    """
    
    results = {'n_qubits': n_qubits}
    
    # ============ STEP 1: Readout Calibration ============
    p0, p1 = 1.0, 1.0
    if use_readout_mitigation:
        if verbose: print("  Calibrating readout errors...")
        p0, p1 = calibrate_readout(backend, n_qubits, shots // 2, noise_model)
        results['readout_p0'] = p0
        results['readout_p1'] = p1
        results['readout_error_rate'] = 1 - (p0 + p1) / 2
        if verbose: print(f"    p(0|0)={p0:.3f}, p(1|1)={p1:.3f}, ε={results['readout_error_rate']:.3f}")
    
    # ============ STEP 2: Prepare Circuits ============
    qc_A_base = make_cluster_state_circuit(n_qubits)
    for i in range(0, n_qubits, 2): qc_A_base.h(i)
    
    qc_B_base = make_cluster_state_circuit(n_qubits)
    for i in range(1, n_qubits, 2): qc_B_base.h(i)
    
    # ============ STEP 3: Execute with optional ZNE ============
    if use_zne:
        if verbose: print("  Running ZNE...")
        executor_A = make_executor_for_zne(backend, shots, n_qubits, 'A', noise_model)
        executor_B = make_executor_for_zne(backend, shots, n_qubits, 'B', noise_model)
        
        factory = zne.inference.LinearFactory(scale_factors=zne_scale_factors)
        
        prob_A = zne.execute_with_zne(qc_A_base, executor_A, factory=factory,
                                       scale_noise=zne.scaling.fold_global)
        prob_B = zne.execute_with_zne(qc_B_base, executor_B, factory=factory,
                                       scale_noise=zne.scaling.fold_global)
    else:
        # Direct execution
        qc_A = qc_A_base.copy()
        qc_A.measure_all()
        qc_B = qc_B_base.copy()
        qc_B.measure_all()
        
        if noise_model:
            result_A = backend.run(transpile(qc_A, backend), shots=shots, noise_model=noise_model).result()
            result_B = backend.run(transpile(qc_B, backend), shots=shots, noise_model=noise_model).result()
        else:
            result_A = backend.run(transpile(qc_A, backend), shots=shots).result()
            result_B = backend.run(transpile(qc_B, backend), shots=shots).result()
        
        counts_A = result_A.get_counts()
        counts_B = result_B.get_counts()
        
        # Compute probabilities
        def compute_prob(counts, indices):
            total = sum(counts.values())
            if total == 0: return 0.0
            success = sum(c for b, c in counts.items() 
                         if all(stabilizer_eigenvalue(b, i, n_qubits) == +1 for i in indices))
            return success / total
        
        prob_A = compute_prob(counts_A, range(0, n_qubits, 2))
        prob_B = compute_prob(counts_B, range(1, n_qubits, 2))
    
    results['prob_A_raw'] = prob_A
    results['prob_B_raw'] = prob_B
    
    # ============ STEP 4: Apply Readout Correction ============
    if use_readout_mitigation:
        prob_A = apply_readout_correction(prob_A, p0, p1)
        prob_B = apply_readout_correction(prob_B, p0, p1)
    
    # Clamp to valid range
    prob_A = max(0.0, min(1.0, prob_A))
    prob_B = max(0.0, min(1.0, prob_B))
    
    results['prob_A'] = prob_A
    results['prob_B'] = prob_B
    
    # ============ STEP 5: Compute Witness ============
    w_value = 3 - 2 * (prob_A + prob_B)
    
    # Error estimation (binomial standard error)
    var_A = prob_A * (1 - prob_A) / shots
    var_B = prob_B * (1 - prob_B) / shots
    se = 2 * np.sqrt(var_A + var_B)
    
    results['w_value'] = w_value
    results['se'] = se
    results['gme_detected'] = w_value < 0
    
    return results

# ============================================================
# NOISE MODEL FOR SIMULATION
# ============================================================

def create_noise_model(single_qubit_error=0.01, two_qubit_error=0.02, readout_error=0.02):
    """Create noise model matching linear_witness.py parameters."""
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(single_qubit_error, 1), ['h'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error, 2), ['cz', 'cx'])
    readout_probs = [[1 - readout_error, readout_error], [readout_error, 1 - readout_error]]
    noise_model.add_all_qubit_readout_error(ReadoutError(readout_probs))
    return noise_model

# ============================================================
# MAIN: CONTRIBUTION BREAKDOWN
# ============================================================

if __name__ == "__main__":
    print("=" * 95)
    print("LINEAR CLUSTER WITNESS - ERROR MITIGATION CONTRIBUTION ANALYSIS")
    print("=" * 95)
    
    print("""
HOW EACH TECHNIQUE WORKS:
─────────────────────────
1. READOUT ERROR MITIGATION (LINEAR)
   - Calibrate: Prepare |0⟩^n and |1⟩^n, measure error rates
   - Correct: p_true = (p_measured - ε) / (1 - 2ε)
   - This is a LINEAR transformation
   - Valid: Standard technique, assumes symmetric errors
   
2. ZERO NOISE EXTRAPOLATION (ZNE)
   - Run at noise levels 1x, 2x, 3x (via gate folding)
   - Fit line to (noise, result) points
   - Extrapolate to 0x noise
   - Valid: Well-established, works when noise is moderate
""")
    
    # Setup
    noise_model = create_noise_model(0.01, 0.02, 0.02)
    backend = AerSimulator(noise_model=noise_model)
    
    print("Noise: 1% single-qubit, 2% two-qubit, 2% readout")
    print("=" * 95)
    
    # Test contribution of each technique
    print("\nCONTRIBUTION BREAKDOWN:")
    print("-" * 75)
    print(f"{'N':>3} | {'Raw':>10} | {'+ REM':>10} | {'+ ZNE':>10} | {'Δ(REM)':>8} | {'Δ(ZNE)':>8} | {'GME?'}")
    print("-" * 75)
    
    for n in [6, 8, 10, 12, 14, 15, 16]:
        # Raw
        r0 = run_witness(n, backend, noise_model=noise_model,
                        use_readout_mitigation=False, use_zne=False)
        
        # + Readout Mitigation (LINEAR)
        r1 = run_witness(n, backend, noise_model=noise_model,
                        use_readout_mitigation=True, use_zne=False)
        
        # + ZNE (full)
        r2 = run_witness(n, backend, noise_model=noise_model,
                        use_readout_mitigation=True, use_zne=True,
                        zne_scale_factors=[1.0, 2.0, 3.0])
        
        # Compute individual contributions
        delta_rem = r0['w_value'] - r1['w_value']  # Readout mitigation contribution (LINEAR)
        delta_zne = r1['w_value'] - r2['w_value']  # ZNE contribution
        
        gme = "YES" if r2['gme_detected'] else "no"
        if r2['gme_detected'] and not r0['gme_detected']:
            gme = "YES! ✓"
        
        print(f"{n:>3} | {r0['w_value']:>+10.3f} | {r1['w_value']:>+10.3f} | {r2['w_value']:>+10.3f} | {delta_rem:>+8.3f} | {delta_zne:>+8.3f} | {gme}")
    
    print("-" * 75)
    print("Δ(REM) = contribution from LINEAR Readout Error Mitigation")
    print("Δ(ZNE) = contribution from Zero Noise Extrapolation")
    
    print("\n" + "=" * 95)
    print("FOR REAL HARDWARE:")
    print("=" * 95)
    print("""
1. Remove noise_model parameter (real hardware has its own noise)
2. Use IQM backend:
   
   from iqm.qiskit_iqm import IQMProvider
   provider = IQMProvider("https://cocos.resonance.meetiqm.com/garnet")
   backend = provider.get_backend()
   
   result = run_witness(n, backend, shots=4000,
                       use_readout_mitigation=True,
                       use_zne=True)

3. ZNE may be slower (3x circuit executions) - consider disabling for initial tests
4. LINEAR Readout mitigation is fast and effective
""")
    print("=" * 95)
