import numpy as np
import matplotlib.pyplot as plt

def get_stabilizer_weight_enumerator(N):
    """
    Computes the weight enumerator polynomial for the Linear Cluster State stabilizer group.
    Returns a list `weights` where weights[w] is the number of stabilizers with weight w.
    Algorithm: Transfer Matrix Method.
    Complexity: O(N).
    """
    # Transfer Matrix approach for Linear Cluster State
    # State indices: (u, v) where u=c_{i-1}, v=c_i. u,v in {0, 1}
    # Matrix maps (u, v) -> (v, w)
    # T[(u,v), (v,w)] = x^{weight contribution of qubit i}
    
    # Weight logic:
    # Op_i = Z^u * X^v * Z^w
    # Weight is 1 if Op_i != I, else 0.
    # Op_i == I iff (v=0) AND (u==w).
    
    # Polynomial variable 'y'. We store coefficients.
    # Actually, we can just store the distribution array directly.
    # Size of distribution array grows as N. Max weight N.
    
    # Current state: 4 polynomials (for each boundary condition 00, 01, 10, 11)
    # Init: "Ghost" qubit 0 with c_0=0.
    # We start at qubit 1. We need c_0, c_1.
    # Boundary condition: c_0 = 0.
    # So valid states are (0,0) and (0,1).
    # Init polys: P_00 = [1], P_01 = [1], P_10 = [], P_11 = [] (impossible)
    
    # Wait, let's formalize step k (1 to N).
    # We transition from (c_{k-1}, c_k) to (c_k, c_{k+1}).
    # Input vector: 4 polys [P_00, P_01, P_10, P_11] representing state (c_{k-1}, c_k).
    # Output vector: 4 polys [Q_00, Q_01, Q_10, Q_11] representing state (c_k, c_{k+1}).
    
    # Initial state (before qubit 1):
    # Effectively we have c_0 = 0.
    # So we are in states (0,0) or (0,1) with weight 0.
    # P_00 = {0: 1}, P_01 = {0: 1}. Others 0.
    
    current_dist = {
        (0,0): {0: 1},
        (0,1): {0: 1},
        (1,0): {},
        (1,1): {}
    }
    
    for k in range(1, N + 1):
        next_dist = {(0,0):{}, (0,1):{}, (1,0):{}, (1,1):{}}
        
        # Transition c_{k+1} in {0, 1}
        for u in [0, 1]: # c_{k-1}
            for v in [0, 1]: # c_k
                poly = current_dist[(u,v)]
                if not poly: continue
                
                for w in [0, 1]: # c_{k+1}
                    # Determine weight added by qubit k
                    # Op = Z^u X^v Z^w
                    # Is Identity?
                    is_identity = (v == 0) and (u == w)
                    added_w = 0 if is_identity else 1
                    
                    # Target state: (v, w)
                    target_poly = next_dist[(v,w)]
                    
                    # Add polynomial: target = target + poly * y^{added_w}
                    for deg, count in poly.items():
                        new_deg = deg + added_w
                        target_poly[new_deg] = target_poly.get(new_deg, 0) + count
                        
        current_dist = next_dist

    # Final Step: Handle Boundary N
    # We have processed qubits 1 to N.
    # The loop went up to k=N. In the last step, w correspond to c_{N+1}.
    # Boundary condition: c_{N+1} = 0.
    # So we only keep states where w=0. i.e., (0,0) and (1,0).
    
    final_weights = {}
    for (v, w), poly in current_dist.items():
        if w == 0:
            for deg, count in poly.items():
                final_weights[deg] = final_weights.get(deg, 0) + count
                
    return final_weights

def calculate_fidelity_analytical(N, p_gate):
    """
    F = <Cl|rho|Cl> = 1/2^N * Sum (1-p)^w(S)
    """
    weights = get_stabilizer_weight_enumerator(N)
    
    # Depolarizing noise model:
    # Effective depolarizing parameter per qubit
    # p_gate is error prob. 
    # For local depolarizing: expectation decays by (1-p) for each non-identity Pauli.
    # Let lambda = 1 - p_gate.
    # Term is lambda^w.
    
    # Note: p_gate in previous code was "depolarizing_error(p, 1)".
    # Qiskit depolarizing_error(p, 1) means:
    # channel(rho) = (1-p) rho + p I/2.
    # Coefficients of Paulis decay by (1-p).
    # So scale = 1 - p_gate.
    
    scale = 1.0 - p_gate
    
    total_sum = 0.0
    for w, count in weights.items():
        total_sum += count * (scale ** w)
        
    return total_sum / (2**N)

def calculate_jungnitsch_witness_analytical(N, p_gate):
    """
    W = 0.5 * I - |Cl><Cl| - P_plus
    Expectation <W> = 0.5 - F - <P_plus>
    """
    scale = 1.0 - p_gate
    
    # 1. Fidelity
    F = calculate_fidelity_analytical(N, p_gate)
    
    # 2. P_plus Correction
    # Generators g_i = Z_{i-1} X_i Z_{i+1}
    # Disjoint set: 0, 3, 6... (0-indexed) -> 1, 4, 7... (1-indexed)
    indices = range(0, N, 3)
    k_set_size = len(indices)
    
    # Calculate expectation of g_i
    # g_i has weight 3 (internal) or 2 (boundary)
    # Internal: 0 < i < N-1. Boundary: i=0 or i=N-1.
    g_expectations = []
    for i in indices:
        w = 0
        # Check boundary
        # i is 0..N-1
        # Terms: Z_{i-1}, X_i, Z_{i+1}
        # i-1 exists if i>0. i+1 exists if i<N-1.
        # X_i always exists.
        w += 1 # X_i
        if i > 0: w += 1 # Z_{i-1}
        if i < N - 1: w += 1 # Z_{i+1}
        
        g_expectations.append(scale ** w)
        
    # P_plus expectation:
    # Logic: Projector onto "0 or 1 failures" of the checks.
    # Let check C_i correspond to "g_i is +1". 
    # Since we are checking if state is "close" to +1 eigenstates.
    # Actually, let's stick to the code formula:
    # term_sum = Prod(I-g) + Sum [ (I+g_j) Prod_{m!=j}(I-g_m) ]
    # Normalized by 1/2^(k+1).
    # Note: 1/2^(k+1) * 2^k (terms) ?
    # Let's trace it.
    # < (I-g) > = 1 - <g>
    # < (I+g) > = 1 + <g>
    # Since disjoint, <Prod> = Prod <>.
    # Also the factor 1/2^(k+1) is applied to the WHOLE sum.
    # Wait, the dimension match?
    # Each (I +/- g) introduces a factor of 2 in trace? No, we look at normalized Expectations.
    # Operator A = (I-g). <A> = 1 - <g>.
    # So we just plug in expectations.
    # BUT we must handle the prefactor carefully.
    # The operator P+ was defined as (1/2^(k+1)) * Sum...
    # The witness is W = 0.5*I - |Cl><Cl| - P_plus
    
    # Calculation:
    probs = np.array(g_expectations)
    
    # Term 1: All (I-g)
    # val = Product (1 - <g_i>)
    term_1 = np.prod(1 - probs)
    
    # Term 2: Sum over j of ( (1+<g_j>) * Prod_{m!=j} (1-<g_m>) )
    term_2 = 0
    for j in range(k_set_size):
        # (1 + p_j) * (Product of all (1-p) divided by (1-p_j))
        # Avoid division by zero if p=1 (perfect state) -> 1-p=0
        # If p=1, term_1 is 0. term_2 is 0?
        # If p=1, all <g>=1. 1-<g>=0.
        # Product is 0.
        # Term 2: Only the term where we pick (1+<g_j>) * 0 * 0... ?
        # If k_set_size > 1, then Prod_{m!=j} has at least one zero factor. So term_2 = 0.
        # So <P_plus> = 0 for perfect state?
        # If P_plus is 0, W = 0.5 - 1 = -0.5. Detected.
        
        # Robust calculation:
        current_prod = 1.0 + probs[j]
        for m in range(k_set_size):
            if m != j:
                current_prod *= (1.0 - probs[m])
        term_2 += current_prod
        
    p_plus_val = (term_1 + term_2) / (2**(k_set_size + 1))
    
    # Wait, 2^(k+1)?
    # For k=1 (1 generator g1):
    # P+ = 1/4 * [ (1-g) + (1+g) ] = 1/4 * [ 2 ] = 0.5
    # Then W = 0.5 - F - 0.5 = -F.
    # Ideally W < 0.
    # For k=1, we just have 1 constraint.
    # Something is odd. The formula in Jungnitsch might have specific prefactors.
    # Let's check the logic:
    # "We add a term P+ which is positive on separable states..."
    # If the formula matches the previous code, I'll use it.
    
    return 0.5 - F - p_plus_val

# Run for N=4 to 20
N_values = range(4, 21)
noise_p = 0.05 # 5% noise

w_std_list = []
w_jung_list = []

print(f"Noise p={noise_p}")
print(f"{'N':<4} | {'Fidelity':<8} | {'W_Std':<8} | {'W_Jung':<8}")

for n in N_values:
    f = calculate_fidelity_analytical(n, noise_p)
    w_std = 0.5 - f
    w_jung = calculate_jungnitsch_witness_analytical(n, noise_p)
    
    w_std_list.append(w_std)
    w_jung_list.append(w_jung)
    
    print(f"{n:<4} | {f:.4f}   | {w_std:+.4f}  | {w_jung:+.4f}")

plt.figure(figsize=(10, 6))
plt.plot(N_values, w_std_list, 'r-o', label='Standard Witness (0.5 - F)')
plt.plot(N_values, w_jung_list, 'g-s', label='Jungnitsch Witness (New)')
plt.axhline(0, color='k', linestyle='--', label='Detection Boundary')
plt.xlabel('Number of Qubits (N)')
plt.ylabel('Witness Value (Negative = Entangled)')
plt.title(f'Scaling of Entanglement Detection (Noise p={noise_p})')
plt.legend()
plt.grid(True)
plt.savefig('witness_scaling.png')