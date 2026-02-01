Let me look at what's actually there and give it to you.OK. The main code is all there. The Qiskit section got cut off but that's just the optional circuit part -- the LP solver and comparison logic are complete. Let me check what actually broke when it ran, then fix it and give it to you.Found the problem. The LP solves but it returns W = +1 with all coefficients zero except the identity. That's the trivial solution -- it satisfies all constraints (W >= 0 on every biseparable state) but it doesn't detect entanglement at all. The objective is minimizing sum(c_S), but the biseparability constraints as currently written are too loose -- they're not actually forcing the witness to be tight against the biseparable set. The sampling approach isn't generating the right constraint geometry.

The issue is that sampling random product states doesn't reliably hit the *extremal* biseparable states that make the constraints binding. Let me rewrite the LP using the exact analytical constraints for linear clusters instead.Good. Now I know exactly what broke and why. The sampling missed the binding constraints entirely -- they live on the Bloch sphere of the shared qubit at each cut. Let me rewrite the LP properly with exact constraints and verify it works before handing it to you.No network. scipy is what we have. Let me check what's available and implement the SDP using scipy's minimize with the correct constraint structure.

bash

cd /home/claude && python3 << 'PYEOF'
"""
The max is simply max(a, b). This makes sense geometrically:
    a*sin(t)*cos(s) + b*cos(t)*sin(s)
At (t=pi/2, s=0): value = a*1*1 + b*0*0 = a
At (t=0, s=pi/2): value = a*0*0 + b*1*1 = b
At any interior point, it's a convex combination, so max is at a vertex = max(a,b).

Wait, that's not a convex combination. Let me check t=s=pi/4:
    a*(sqrt(2)/2)*(sqrt(2)/2) + b*(sqrt(2)/2)*(sqrt(2)/2) = (a+b)/2
For a=b=1: (1+1)/2 = 1 = max(1,1). OK so interior can equal the max but not exceed it.

Actually the function IS bounded by max(a,b) because:
    a*sin(t)*cos(s) <= a*sin(t) <= a  (since cos(s) <= 1)
    b*cos(t)*sin(s) <= b*sin(s) <= b
But a*sin(t)*cos(s) + b*cos(t)*sin(s) <= a + b in general... 
yet numerically it's max(a,b). Let me think again.

sin(t)*cos(s) + cos(t)*sin(s) = sin(t+s) <= 1. So if a=b:
    a*(sin(t)*cos(s) + cos(t)*sin(s)) = a*sin(t+s) <= a = max(a,a). Confirmed.

For a != b, say a > b:
    a*sin(t)*cos(s) + b*cos(t)*sin(s) <= a*sin(t)*cos(s) + a*cos(t)*sin(s) = a*sin(t+s) <= a.
    And equals a at t=pi/2, s=0. So max = a = max(a,b). QED.

Great. So the biseparability constraint for the generator-only witness at cut j is:

    W_min across cut j = c_0 - sum_{k != j-1, j} |c_k| - max(|c_{j-1}|, |c_j|) >= 0

Now: this is a LINEAR constraint in c (with absolute values). We can linearize
by introducing auxiliary variables or by considering all sign combinations.

For the objective, minimize c_0 + sum(c_k) on the target state.

Note: on the target state, all <g_k> = +1, so we want c_k < 0 to make W negative.
The constraint says c_0 must be large enough to keep W non-negative on bisep states.

Let's parameterize: let d_k = -c_k >= 0 for k=0..N-1 (the negative coefficients),
and c_0 > 0 (the identity term).

Objective: minimize c_0 - sum(d_k)  [want this as negative as possible, i.e., maximize sum(d_k) - c_0]

Constraint at cut j:
    c_0 - sum_{k != j-1, j} d_k - max(d_{j-1}, d_j) >= 0
    => c_0 >= sum_{k != j-1, j} d_k + max(d_{j-1}, d_j)
    => c_0 >= sum_{k} d_k - min(d_{j-1}, d_j)
       [since max(a,b) = a+b - min(a,b), so sum_other + max = sum_all - min]

So: c_0 >= sum(d_k) - min(d_{j-1}, d_j) for each j = 1..N-1.

The tightest constraint is the one with the SMALLEST min(d_{j-1}, d_j),
i.e., the weakest link in the chain.

Objective: minimize c_0 - sum(d_k). With c_0 = max over j of [sum(d_k) - min(d_{j-1}, d_j)]:
    W_target = max_j [sum(d_k) - min(d_{j-1}, d_j)] - sum(d_k)
             = -min_j [min(d_{j-1}, d_j)]
             = -min over all adjacent pairs (d_{j-1}, d_j) of min(d_{j-1}, d_j)
             = -min over all k of d_k  (since each d_k appears in at least one adjacent pair)

Wait -- min_j min(d_{j-1}, d_j) = min over all k of d_k (since every k except 0 and N-1
appears as both d_{j-1} and d_j for some j, and d_0 appears as d_{j-1} for j=1,
and d_{N-1} appears as d_j for j=N-1).

So W_target = -min_k(d_k). To minimize W_target, we maximize min_k(d_k).
The optimal solution has ALL d_k EQUAL: d_k = d for all k.

Then c_0 = N*d - d = (N-1)*d, and W_target = c_0 - N*d = (N-1)*d - N*d = -d.

We have one free parameter d > 0 (overall scale). The witness is:
    W = (N-1)*d * I - d * sum_k g_k = d * [(N-1)*I - sum_k g_k]

The scale doesn't matter for entanglement detection (W < 0 iff the bracket is < 0).
So the generator-only optimized witness is just:
    W = (N-1)*I - sum_{k=0}^{N-1} g_k

This is actually the SAME as the generic witness! (Toth & Guhne Theorem 2 gives
exactly this: W = (N-1)I - sum g_k, which equals 3I - 2(P_even + P_odd) when expanded.)

So: optimizing over generator coefficients alone gives back the generic witness.
The improvement MUST come from including stabilizer PRODUCTS (multi-generator elements).
"""

import numpy as np

# Verify: (N-1)I - sum g_k equals the Toth & Guhne witness
# Toth & Guhne: W = 3I - 2*(P_even + P_odd)
# P_even = prod_{k even} (I + g_k)/2
# P_odd  = prod_{k odd}  (I + g_k)/2
# 
# For N=4: P_even = (I+g_0)(I+g_2)/4 = (I + g_0 + g_2 + g_0*g_2)/4
#           P_odd  = (I+g_1)(I+g_3)/4 = (I + g_1 + g_3 + g_1*g_3)/4
# W = 3I - 2*(P_even + P_odd)
#   = 3I - (I + g_0 + g_2 + g_0g_2)/2 - (I + g_1 + g_3 + g_1g_3)/2
#   = 3I - I - (g_0+g_1+g_2+g_3)/2 - (g_0g_2 + g_1g_3)/2
#   = 2I - (sum g_k)/2 - (g_0g_2 + g_1g_3)/2
#
# That's NOT (N-1)I - sum g_k = 3I - sum g_k.
# So Toth & Guhne includes the product terms g_0*g_2 and g_1*g_3!
# Those are the within-color-class products.
# And (N-1)I - sum g_k is a DIFFERENT (weaker) witness.

# So the generic Toth & Guhne witness already includes within-color products.
# The Jungnitsch optimization adds DIFFERENT WEIGHTS to those products.
# That's where the improvement comes from.

# Let me redo the optimization including within-color products.
# For a linear cluster, color class A = {g_0, g_2, g_4, ...}
# Within-color products: all subsets of A, and all subsets of B.
# These are exactly the elements measurable in the two-setting protocol.

# Number of variables: 2^|A| + 2^|B| - 1 (subtract 1 for shared identity)
# For N=20: |A|=10, |B|=10, variables = 2^10 + 2^10 - 1 = 2047.

# The biseparability constraint analysis needs to account for products too.
# For a product g_a * g_b (both in color A, both on side A of cut j),
# <g_a * g_b> = <g_a> * <g_b> on ANY state (they commute and are independent 
# on a product state... wait, are they independent?)

# g_a and g_b share qubits if they're adjacent in the chain.
# g_0 = X_0 Z_1, g_2 = Z_1 X_2 Z_3. They share qubit 1 (Z in both).
# <g_0 * g_2> = <X_0 Z_1 * Z_1 X_2 Z_3> = <X_0 X_2 Z_3> (Z_1^2 = I)
# On a product state: <X_0>*<X_2>*<Z_3>, all free. So <g_0*g_2> is free (+/-1).

# In general, for within-color products, the shared Z's cancel (Z^2 = I),
# leaving a product of X's and boundary Z's -- all on different qubits.
# So within-color products are ALWAYS free (+/-1) on product states.

# This means the biseparability analysis for within-color products is trivial:
# each product element can independently be +/-1 on bisep states,
# EXCEPT for the crossing products that involve generators on both sides of the cut.

# For cut j: generators g_{j-1} and g_j cross. Color of g_{j-1}: (j-1)%2.
# Color of g_j: j%2. These are DIFFERENT colors (adjacent generators alternate color).
# So no within-color product contains BOTH g_{j-1} and g_j.

# Products containing g_{j-1} (color (j-1)%2): their expectation involves the
# shared-qubit constraint at qubit j-1 (same as before for single generator).
# Products containing g_j: involve shared-qubit constraint at qubit j.
# Products containing neither: entirely on one side, free (+/-1).

# So the constraint structure is:
# For cut j, partition all stabilizer elements into:
#   Group F (free): elements not containing g_{j-1} or g_j -> expectation is free +/-1
#   Group L (left-crossing): elements containing g_{j-1} but not g_j
#   Group R (right-crossing): elements containing g_j but not g_{j-1}
#
# Group L elements' expectations are proportional to <g_{j-1}> (the other generators
# in the product contribute free +/-1 signs). So <S> = (+/-1) * |<g_{j-1}>|'s amplitude.
# Similarly Group R ~ |<g_j}>|'s amplitude.
#
# And we showed max over (t,s) of |amplitude_{j-1}|*a + |amplitude_j|*b = max(a,b)
# where a = max contribution from L group, b = max from R group.
#
# Wait, this isn't quite right because each L element has its OWN sign choice
# independent of other L elements. Let me reconsider.

# For element S in group L (contains g_{j-1}, not g_j):
# S = g_{j-1} * (product of other generators, all in same color class, not g_j)
# <S> on product state = <g_{j-1}> * product of <g_k> for other k in S
# The other <g_k> are free +/-1 (their generators don't cross cut j).
# <g_{j-1}> itself has amplitude sin(t)*cos(s) (from shared qubit analysis).
# So <S> = (+/-1) * sin(t)*cos(s) for each S in group L.
# The sign is chosen INDEPENDENTLY for each S.

# Similarly for group R: <S> = (+/-1) * cos(t)*sin(s)

# So W on a bisep state across cut j:
# W = sum_{S in F} c_S * eps_S + sum_{S in L} c_S * eps_S * sin(t)*cos(s) 
#   + sum_{S in R} c_S * eps_S * cos(t)*sin(s)
# where each eps_S is independently +/-1.

# Minimum over all eps choices and (t,s):
# W_min = -sum_{F} |c_S| - (sum_{L} |c_S|)*max_{t,s} sin(t)*cos(s) ... 
# NO. The eps's and (t,s) interact. For given (t,s), optimal eps gives:
# W_min(t,s) = -sum_F |c_S| - sin(t)*cos(s) * sum_L |c_S| - cos(t)*sin(s) * sum_R |c_S|

# Then minimize over (t,s):
# W_min = -sum_F |c_S| - max_{t,s} [A*sin(t)*cos(s) + B*cos(t)*sin(s)]
# where A = sum_L |c_S|, B = sum_R |c_S|
# And we showed this max = max(A, B).

# So: W_min across cut j = -sum_F |c_S| - max(sum_L |c_S|, sum_R |c_S|) >= 0
# Plus the identity term c_I.

# Constraint: c_I >= sum_F |c_S| + max(sum_L |c_S|, sum_R |c_S|)

# This is EXACT (not an approximation) for the two-setting witness with 
# within-color-class stabilizer elements only.

print("Exact biseparability constraint derived.")
print("Constraint at cut j: c_I >= F_j + max(L_j, R_j)")
print("where F_j = sum |c_S| for free elements")
print("      L_j = sum |c_S| for left-crossing elements")  
print("      R_j = sum |c_S| for right-crossing elements")
print()
print("This can be converted to a LINEAR PROGRAM by noting c_S < 0 for all S != I")
print("(we want negative coefficients to make W < 0 on target state).")
print("Let d_S = -c_S >= 0. Then |c_S| = d_S.")
print()
print("LP:")
print("  Variables: d_S >= 0 for each non-identity stabilizer element S")  
print("  Objective: maximize sum(d_S) - c_I  [= -W_target, want this positive]")
print("  Constraints: c_I >= F_j + max(L_j, R_j) for each j")
print("  Equivalently: c_I >= F_j + L_j and c_I >= F_j + R_j for each j")
print("  (since max(a,b) >= a and max(a,b) >= b, and the binding one is max)")
print()
print("This is a clean LP. Let me implement it.")

PYEOF
Output

Exact biseparability constraint derived.
Constraint at cut j: c_I >= F_j + max(L_j, R_j)
where F_j = sum |c_S| for free elements
      L_j = sum |c_S| for left-crossing elements
      R_j = sum |c_S| for right-crossing elements

This can be converted to a LINEAR PROGRAM by noting c_S < 0 for all S != I
(we want negative coefficients to make W < 0 on target state).
Let d_S = -c_S >= 0. Then |c_S| = d_S.

LP:
  Variables: d_S >= 0 for each non-identity stabilizer element S
  Objective: maximize sum(d_S) - c_I  [= -W_target, want this positive]
  Constraints: c_I >= F_j + max(L_j, R_j) for each j
  Equivalently: c_I >= F_j + L_j and c_I >= F_j + R_j for each j
  (since max(a,b) >= a and max(a,b) >= b, and the binding one is max)

This is a clean LP. Let me implement it.

cd /home/claude && python3 << 'PYEOF'
import numpy as np
from scipy.optimize import linprog
from itertools import combinations

def solve_optimized_witness(N, verbose=True):
    """
    Exact LP for optimized entanglement witness on N-qubit linear cluster state.
    
    Stabilizer elements: all products of subsets of generators within a single
    color class (the two-setting measurable elements).
    
    Color A (even generators): g_0, g_2, g_4, ...  -> ceil(N/2) generators
    Color B (odd generators):  g_1, g_3, g_5, ...  -> floor(N/2) generators
    
    Elements: all 2^|A| subsets of A generators, plus all 2^|B| subsets of B generators,
    minus 1 for the shared identity.
    
    Variables: d_S >= 0 for each non-identity element (d_S = -c_S, the positive weight).
               c_I = identity coefficient (positive, determined by constraints).
    
    Objective: minimize W_target = c_I - sum(d_S)
               equivalently maximize sum(d_S) - c_I
    
    Constraints: for each cut j = 1..N-1:
        c_I >= sum_F d_S + sum_L d_S    (left-crossing constraint)
        c_I >= sum_F d_S + sum_R d_S    (right-crossing constraint)
    
    where at cut j:
        F = elements containing neither g_{j-1} nor g_j  (free, +/-1 on bisep)
        L = elements containing g_{j-1} but not g_j       (left-crossing)
        R = elements containing g_j but not g_{j-1}       (right-crossing)
    
    Note: g_{j-1} and g_j are in DIFFERENT color classes (adjacent generators
    alternate color), so no element contains both. Thus F, L, R partition
    all non-identity elements.
    """
    
    # --- Enumerate elements ---
    even_gens = list(range(0, N, 2))   # g_0, g_2, g_4, ...
    odd_gens  = list(range(1, N, 2))   # g_1, g_3, g_5, ...
    
    # All non-empty subsets of even_gens (color A elements)
    color_A_elements = []
    for r in range(1, len(even_gens) + 1):
        for subset in combinations(even_gens, r):
            color_A_elements.append(frozenset(subset))
    
    # All non-empty subsets of odd_gens (color B elements)
    color_B_elements = []
    for r in range(1, len(odd_gens) + 1):
        for subset in combinations(odd_gens, r):
            color_B_elements.append(frozenset(subset))
    
    # Full element list (non-identity)
    all_elements = color_A_elements + color_B_elements
    n_elem = len(all_elements)  # 2^|A| - 1 + 2^|B| - 1
    
    elem_to_idx = {elem: i for i, elem in enumerate(all_elements)}
    
    if verbose:
        print(f"N={N}: {len(even_gens)} even gens, {len(odd_gens)} odd gens, "
              f"{len(color_A_elements)} color-A elements, {len(color_B_elements)} color-B elements, "
              f"{n_elem} total non-identity elements")
    
    # --- Build LP ---
    # Variables: [d_0, d_1, ..., d_{n_elem-1}, c_I]
    # d_S >= 0 (enforced by bounds), c_I free (but will be positive at optimum)
    n_vars = n_elem + 1  # d's + c_I
    c_I_idx = n_elem     # index of c_I in variable vector
    
    # Objective: minimize c_I - sum(d_S) = minimize [-d_0, -d_1, ..., -d_{n-1}, +1] . x
    c_obj = np.zeros(n_vars)
    c_obj[:n_elem] = -1.0  # -d_S terms
    c_obj[c_I_idx] = 1.0   # +c_I term
    
    # Constraints: for each cut j, two inequalities:
    #   c_I - sum_F d_S - sum_L d_S >= 0  ->  -c_I + sum_F d_S + sum_L d_S <= 0
    #   c_I - sum_F d_S - sum_R d_S >= 0  ->  -c_I + sum_F d_S + sum_R d_S <= 0
    
    A_ub_rows = []
    b_ub_rows = []
    
    for j in range(1, N):  # cuts at j = 1, 2, ..., N-1
        # g_{j-1} and g_j are the crossing generators
        g_left  = j - 1  # crosses from A side
        g_right = j      # crosses from B side
        
        # Classify elements
        # L: contains g_left, not g_right
        # R: contains g_right, not g_left  
        # F: contains neither
        # (No element contains both since they're different colors)
        
        # Left constraint: c_I >= F + L  ->  -c_I + F + L <= 0
        row_L = np.zeros(n_vars)
        row_L[c_I_idx] = -1.0
        for i, elem in enumerate(all_elements):
            if g_left in elem or g_right not in elem:
                # This element is in F or L (not in R)
                # F: neither g_left nor g_right -> contributes to both constraints
                # L: has g_left -> contributes to left constraint only
                if g_right not in elem:  # F or L
                    row_L[i] = 1.0
        A_ub_rows.append(row_L)
        b_ub_rows.append(0.0)
        
        # Right constraint: c_I >= F + R  ->  -c_I + F + R <= 0
        row_R = np.zeros(n_vars)
        row_R[c_I_idx] = -1.0
        for i, elem in enumerate(all_elements):
            if g_left not in elem:  # F or R
                row_R[i] = 1.0
        A_ub_rows.append(row_R)
        b_ub_rows.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    # Bounds: d_S >= 0, c_I unbounded (but will be positive)
    bounds = [(0, None)] * n_elem + [(None, None)]
    
    # Need a normalization constraint to prevent trivial d=0 solution.
    # Fix c_I = 1 (WLOG, witness is scale-invariant for detection).
    A_eq = np.zeros((1, n_vars))
    A_eq[0, c_I_idx] = 1.0
    b_eq = np.array([1.0])
    
    # Solve
    result = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')
    
    if not result.success:
        print(f"  LP FAILED: {result.message}")
        return None
    
    # Extract solution
    d_values = result.x[:n_elem]
    c_I_val = result.x[c_I_idx]
    
    # Coefficients: c_S = -d_S for non-identity, c_I for identity
    coefficients = {frozenset(): c_I_val}
    for i, elem in enumerate(all_elements):
        coefficients[elem] = -d_values[i]
    
    # W on target state = c_I - sum(d_S)
    w_target = c_I_val - np.sum(d_values)
    
    # --- Compute noise tolerance ---
    # Under white noise parameter p: rho = (1-p)|psi><psi| + p*I/2^N
    # <S>_noisy = (1-p)^weight(S) for non-identity S (weight = number of qubits acted on)
    # <I>_noisy = 1 always
    # W(p) = c_I + sum_S c_S * (1-p)^weight(S)
    
    # Compute weights
    def stabilizer_weight(elem):
        """Weight = number of qubits with non-trivial Pauli in the product."""
        x_count = np.zeros(N, dtype=int)
        z_count = np.zeros(N, dtype=int)
        for idx in elem:
            if idx > 0: z_count[idx-1] = (z_count[idx-1] + 1) % 2
            x_count[idx] = (x_count[idx] + 1) % 2
            if idx < N-1: z_count[idx+1] = (z_count[idx+1] + 1) % 2
        return int(np.sum((x_count + z_count) > 0))
    
    weights = {elem: stabilizer_weight(elem) for elem in all_elements}
    
    def witness_under_noise(p):
        w = c_I_val  # identity term
        for i, elem in enumerate(all_elements):
            w += (-d_values[i]) * ((1-p) ** weights[elem])
        return w
    
    # Binary search for p_crit (W(p) = 0)
    p_lo, p_hi = 0.0, 1.0
    if witness_under_noise(0.0) >= 0:
        # Witness doesn't even detect at p=0
        noise_tol = 0.0
    else:
        for _ in range(60):
            p_mid = (p_lo + p_hi) / 2
            if witness_under_noise(p_mid) < 0:
                p_lo = p_mid
            else:
                p_hi = p_mid
        noise_tol = (p_lo + p_hi) / 2
    
    # Generic Toth & Guhne noise tolerance for comparison
    # The generic witness has P_even and P_odd each decaying as (1-p)^{weight of product}.
    # For large N, the tolerance approaches 1 - 1/sqrt(2) ~ 0.293 ... 
    # Actually for the two-projector witness it's more complex. Let me just compute
    # it by evaluating the generic witness under noise.
    
    def generic_witness_noise(p):
        """Generic Toth & Guhne: W = (N-1)I - 2*(P_even + P_odd) expanded."""
        # P


cd /home/claude && python3 << 'EOF'
import numpy as np
from scipy.optimize import linprog
from itertools import combinations, product as iterproduct

def linear_cluster_stabilizers(N):
    """Generator g_k: X on k, Z on neighbors."""
    stabilizers = []
    for k in range(N):
        x_set = {k}
        z_set = set()
        if k > 0: z_set.add(k - 1)
        if k < N - 1: z_set.add(k + 1)
        stabilizers.append((x_set, z_set))
    return stabilizers

def all_stabilizer_elements(N):
    """All 2^N products of subsets of generators."""
    elements = []
    for r in range(N + 1):
        for subset in combinations(range(N), r):
            elements.append(frozenset(subset))
    return elements

def get_pauli_string(N, stabs, element):
    """
    Returns the Pauli string for a stabilizer element as two arrays:
    x_vec[q] = 1 if X or Y acts on qubit q, else 0
    z_vec[q] = 1 if Z or Y acts on qubit q, else 0
    (mod 2 arithmetic since generators square to I)
    """
    x_vec = np.zeros(N, dtype=int)
    z_vec = np.zeros(N, dtype=int)
    for idx in element:
        x_set, z_set = stabs[idx]
        for q in x_set: x_vec[q] = (x_vec[q] + 1) % 2
        for q in z_set: z_vec[q] = (z_vec[q] + 1) % 2
    return x_vec, z_vec

def build_bisep_constraints(N):
    """
    For each cut k (A={0..k-1}, B={k..N-1}), enumerate the extremal
    biseparable constraints.

    At cut k, generators g_{k-1} and g_k both cross. They share qubit k-1
    on side A (g_{k-1} has X there, g_k has Z there) and share qubit k on
    side B (g_{k-1} has Z there, g_k has X there).

    On a product state, the expectation of any stabilizer element S factors as:
        <S> = product over qubits q of <P_q>
    where P_q is the single-qubit Pauli on qubit q in S.

    For qubits not shared between crossing generators, we can set <P_q> = +/-1
    freely (choose eigenstate). For the shared qubits (k-1 and k), the
    expectations are constrained by the Bloch sphere:
        qubit k-1: <X> = x1, <Z> = z1,  x1^2 + z1^2 <= 1
        qubit k:   <X> = x2, <Z> = z2,  x2^2 + z2^2 <= 1

    We discretize each Bloch sphere and enumerate sign choices for free qubits
    to get the binding constraints.
    """
    stabs = linear_cluster_stabilizers(N)
    elements = all_stabilizer_elements(N)
    elem_list = list(elements)
    n_elem = len(elem_list)

    # Precompute Pauli strings for all elements
    pauli_strings = {}
    for elem in elem_list:
        pauli_strings[elem] = get_pauli_string(N, stabs, elem)

    A_ub_rows = []

    # Discretization of Bloch sphere (theta, phi for each shared qubit)
    n_theta = 20
    n_phi = 20
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    for cut_k in range(1, N):
        # Shared qubits: k-1 (on side A) and k (on side B)
        # qubit k-1: g_{k-1} has X, g_k has Z
        # qubit k:   g_{k-1} has Z, g_k has X
        shared_A = cut_k - 1  # qubit on A side shared by crossing gens
        shared_B = cut_k      # qubit on B side shared by crossing gens

        # Free qubits: everything except shared_A and shared_B
        # On free qubits, any single Pauli expectation can be +1 or -1.
        # For each element, identify which free qubits it acts on non-trivially.
        # The sign contribution from free qubits is the product of their signs,
        # each chosen to be +1 or -1.
        # To get the MOST POSITIVE W (worst case for the witness), for each
        # element we want sign(free part) * c_S to be as large as possible.
        # Since we're looking for W >= 0 constraints, the worst case is
        # max W over biseparable states. For the LP we need W >= 0 for ALL
        # biseparable states, so we need the constraints at the MAXIMUM.
        #
        # For free qubits, the optimal sign choice is: sign(free_q) = +1 if
        # c_S > 0, else -1. But c_S is what we're solving for, so we can't
        # do this directly.
        #
        # Instead: for each sign assignment on free qubits (2^n_free possibilities),
        # add the constraint. But n_free can be large.
        #
        # Smarter: the free qubits only contribute a sign flip to each element.
        # Group elements by their shared-qubit Pauli content. Elements with the
        # same Pauli on shared qubits but different Pauli on free qubits differ
        # only by a sign that can be flipped independently.
        #
        # Actually, the simplest correct approach: for each element, the free-qubit
        # contribution is +/-1 and can be chosen independently per element (since
        # different elements act on different free qubits, and each free qubit's
        # sign is independent). Wait -- that's not true. Two elements might share
        # a free qubit, and they must use the SAME sign for that qubit.
        #
        # This is getting complicated. Let me use a different approach:
        # enumerate sign assignments for the GENERATORS (not elements), since
        # element signs are determined by generator signs multiplicatively.
        #
        # For generators entirely on side A (g_0, ..., g_{k-2}): sign is free
        # For generators entirely on side B (g_{k+1}, ..., g_{N-1}): sign is free
        # For g_{k-1}: sign = <X_{k-1} Z_{k-2}>_A * <Z_k>_B
        #            = <X_{k-1}> * <Z_{k-2}> * <Z_k>
        #   where <X_{k-1}> comes from shared qubit, <Z_{k-2}> is free (+/-1),
        #   <Z_k> comes from shared qubit k
        # For g_k: sign = <Z_{k-1}>_A * <X_k Z_{k+1}>_B
        #        = <Z_{k-1}> * <X_k> * <Z_{k+1}>
        #   where <Z_{k-1}> comes from shared qubit, <X_k> from shared qubit k,
        #   <Z_{k+1}> is free (+/-1)
        #
        # So generator signs are:
        #   g_j for j < k-1: free sign s_j in {-1, +1}
        #   g_{k-1}: sign = x1 * s_{k-2} * z2  (x1=<X> on qubit k-1, z2=<Z> on qubit k)
        #            (if k-1 == 0, no Z_{k-2} term, so sign = x1 * z2)
        #   g_k:     sign = z1 * x2 * s_{k+1}  (z1=<Z> on qubit k-1, x2=<X> on qubit k)
        #            (if k == N-1, no Z_{k+1} term, so sign = z1 * x2)
        #   g_j for j > k: free sign s_j in {-1, +1}
        #
        # Element signs = product of constituent generator signs.
        # So element signs are determined by: (x1, z1, x2, z2) on shared qubits
        # and free signs {s_j} on non-crossing generators.

        # Number of free generators
        free_gens_A = list(range(0, cut_k - 1))  # g_0 ... g_{k-2}
        free_gens_B = list(range(cut_k + 1, N))  # g_{k+1} ... g_{N-1}
        free_gens = free_gens_A + free_gens_B
        n_free = len(free_gens)

        # For each Bloch sphere point and each free-sign assignment, compute
        # the expectation vector and add constraint.
        # Total constraints per cut: n_theta * n_phi^2 * 2^n_free
        # For N=20, cut at k=10: n_free = 18, 2^18 = 262144. Too many.
        #
        # Key insight: for the LP, we only need the EXTREMAL constraints.
        # The extremal product states have all free qubits in eigenstates,
        # i.e., each free generator has sign exactly +1 or -1.
        # So we DO need to enumerate 2^n_free sign assignments.
        # But 2^18 * 400 (Bloch points) = 100M constraints for N=20. Impractical.
        #
        # Better: use the fact that for fixed (x1,z1,x2,z2), the optimal
        # free-sign assignment is determined by the current c_S values.
        # This suggests an iterative approach or cutting-plane method.
        # But for now, let's just verify correctness at small N.

        if n_free > 12:
            print(f"  Cut {cut_k}: {n_free} free generators, skipping (too many)")
            continue

        for t1 in thetas:
            x1 = np.sin(t1)   # <X> on qubit k-1
            z1 = np.cos(t1)   # <Z> on qubit k-1
            for t2 in thetas:
                x2 = np.sin(t2)   # <X> on qubit k
                z2 = np.cos(t2)   # <Z> on qubit k

                # Compute crossing generator signs
                # g_{k-1} sign: x1 * (s_{k-2} if k>1 else 1) * z2
                # g_k sign:     z1 * x2 * (s_{k+1} if k<N-1 else 1)
                # But s_{k-2} and s_{k+1} are free signs -- they're part of the
                # free sign enumeration. Let me restructure.
                #
                # Actually g_{k-2} is a free generator with its own sign s_{k-2}.
                # The Z_{k-2} in g_{k-1} is NOT the same as the sign of g_{k-2}.
                # g_{k-1}'s sign depends on the actual qubit state, not on g_{k-2}'s sign.
                #
                # Let me re-derive. For a product state |psi> = |psi_0>|psi_1>...|psi_{N-1}>:
                # <g_{k-1}> = <psi_{k-2}|Z|psi_{k-2}> * <psi_{k-1}|X|psi_{k-1}> * <psi_k|Z|psi_k>
                #           = z_{k-2} * x_{k-1} * z_k
                # where z_q = <Z>_q and x_q = <X>_q for each qubit.
                #
                # For free qubits (not shared), we can set x_q or z_q to +/-1
                # by choosing eigenstates. But a qubit can only be in ONE state,
                # so if a qubit appears with X in one generator and Z in another,
                # we have a Bloch sphere constraint.
                #
                # Which qubits are "shared" (appear with different Paulis in different gens)?
                # For linear cluster: g_j has X on j and Z on j-1, j+1.
                # Qubit q appears as: X in g_q, Z in g_{q-1} and g_{q+1}.
                # So EVERY interior qubit is shared between 3 generators (X in one, Z in two).
                # The shared qubit constraint applies to ALL interior qubits, not just k-1 and k!

                pass  # This approach is getting unwieldy

        # Let me step back and think about this more carefully.
        break
    
    return None

# Actually, the right realization: EVERY qubit except the endpoints is shared
# between generators. The Bloch sphere constraint applies everywhere.
# This means we can't just enumerate signs -- we need to optimize over
# the full product of Bloch spheres.
#
# But Jungnitsch et al. actually solved this. Let me re-read their construction.
# Their key result (PRA 84, 032310, Theorem 1) is that for graph states,
# the biseparability constraint at cut k reduces to:
#
#   sum_S c_S * <S>_bisep >= 0
#
# where <S>_bisep is computed on the WORST-CASE product state.
# For graph states, they show this can be computed as an SDP (not LP).
#
# The LP relaxation they use is DIFFERENT from what I was trying.
# Let me implement their actual algorithm.

print("Need to implement Jungnitsch SDP formulation directly, not the sampling LP.")
print("The biseparability constraints are semidefinite, not linear.")
print("Will rewrite using cvxpy SDP.")
EOF
Output

Need to implement Jungnitsch SDP formulation directly, not the sampling LP.
The biseparability constraints are semidefinite, not linear.
Will rewrite using cvxpy SDP.