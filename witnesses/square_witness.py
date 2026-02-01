import numpy as np
from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from iqm.qiskit_iqm.iqm_provider import IQMProvider

def get_neighbors_2d(row, col, grid_size):
    """Get the neighbors of a qubit in a 2D square grid."""
    neighbors = []
    # Up
    if row > 0:
        neighbors.append((row - 1, col))
    # Down
    if row < grid_size - 1:
        neighbors.append((row + 1, col))
    # Left
    if col > 0:
        neighbors.append((row, col - 1))
    # Right
    if col < grid_size - 1:
        neighbors.append((row, col + 1))
    return neighbors

def coord_to_index(row, col, grid_size):
    """Convert 2D grid coordinates to 1D qubit index."""
    return row * grid_size + col

def index_to_coord(index, grid_size):
    """Convert 1D qubit index to 2D grid coordinates."""
    return index // grid_size, index % grid_size

def create_square_graph_state(grid_size):
    """
    Create a square graph state on a grid_size x grid_size lattice.
    Graph states are created by:
    1. Initialize all qubits to |+> (Hadamard on |0>)
    2. Apply CZ gates between all adjacent qubits
    """
    n_qubits = grid_size * grid_size
    qc = QuantumCircuit(n_qubits)
    
    # Step 1: Initialize all qubits to |+>
    qc.h(range(n_qubits))
    
    # Step 2: Apply CZ gates between adjacent qubits
    for row in range(grid_size):
        for col in range(grid_size):
            qubit_idx = coord_to_index(row, col, grid_size)
            
            # Apply CZ to right neighbor
            if col < grid_size - 1:
                right_idx = coord_to_index(row, col + 1, grid_size)
                qc.cz(qubit_idx, right_idx)
            
            # Apply CZ to bottom neighbor
            if row < grid_size - 1:
                bottom_idx = coord_to_index(row + 1, col, grid_size)
                qc.cz(qubit_idx, bottom_idx)
    
    return qc

def get_checkerboard_sets(grid_size):
    """
    Get two sets of qubits in a checkerboard pattern.
    Set A: "white" squares (row + col is even)
    Set B: "black" squares (row + col is odd)
    """
    set_A = []  # White squares
    set_B = []  # Black squares
    
    for row in range(grid_size):
        for col in range(grid_size):
            qubit_idx = coord_to_index(row, col, grid_size)
            if (row + col) % 2 == 0:
                set_A.append(qubit_idx)
            else:
                set_B.append(qubit_idx)
    
    return set_A, set_B

def check_stabilizer_2d(bitstring, qubit_idx, grid_size, n_qubits):
    """
    Check if a stabilizer is satisfied for a given qubit in a 2D grid.
    Stabilizer: S_i = X_i * Z_{neighbors}
    Returns +1 if satisfied, -1 if not.
    """
    row, col = index_to_coord(qubit_idx, grid_size)
    neighbors = get_neighbors_2d(row, col, grid_size)
    
    parity = 1
    
    # Center qubit (X measurement)
    if bitstring[n_qubits - 1 - qubit_idx] == '1':
        parity *= -1
    
    # Neighbors (Z measurements)
    for neighbor_row, neighbor_col in neighbors:
        neighbor_idx = coord_to_index(neighbor_row, neighbor_col, grid_size)
        if bitstring[n_qubits - 1 - neighbor_idx] == '1':
            parity *= -1
    
    return parity

def run_square_witness(grid_size, backend, shots=2000):
    """
    Implements a witness for square graph states using checkerboard measurement.
    
    For a square lattice graph state, we use checkerboard coloring:
    - Setting A: Measure stabilizers on "white" squares (row+col even)
    - Setting B: Measure stabilizers on "black" squares (row+col odd)
    
    Formula: W = 3 - 2 * (P_A + P_B)
    Target: W < 0 implies Genuine Multipartite Entanglement
    """
    n_qubits = grid_size * grid_size
    
    # --- 1. CREATE THE GRAPH STATE ---
    qc_base = create_square_graph_state(grid_size)
    
    # --- 2. GET CHECKERBOARD SETS ---
    set_A, set_B = get_checkerboard_sets(grid_size)
    
    # --- 3. DEFINE MEASUREMENT SETTINGS ---
    
    # Setting A: Measure X on white squares, Z on all others
    qc_A = qc_base.copy()
    for qubit_idx in set_A:
        qc_A.h(qubit_idx)  # H before measurement converts Z-basis to X-basis
    qc_A.measure_all()
    
    # Setting B: Measure X on black squares, Z on all others
    qc_B = qc_base.copy()
    for qubit_idx in set_B:
        qc_B.h(qubit_idx)
    qc_B.measure_all()
    
    # --- 4. RUN EXPERIMENTS ---
    print(f"Running Square Witness for {grid_size}x{grid_size} grid ({n_qubits} qubits)...")
    
    job_A = backend.run(transpile(qc_A, backend), shots=shots)
    job_B = backend.run(transpile(qc_B, backend), shots=shots)
    counts_A = job_A.result().get_counts()
    counts_B = job_B.result().get_counts()
    
    # --- 5. CALCULATE PROBABILITIES ---
    
    # Probability that all white square stabilizers are satisfied
    success_count_A = 0
    for bitstring, count in counts_A.items():
        all_satisfied = True
        for qubit_idx in set_A:
            if check_stabilizer_2d(bitstring, qubit_idx, grid_size, n_qubits) == -1:
                all_satisfied = False
                break
        if all_satisfied:
            success_count_A += count
    
    prob_A = success_count_A / shots
    
    # Probability that all black square stabilizers are satisfied
    success_count_B = 0
    for bitstring, count in counts_B.items():
        all_satisfied = True
        for qubit_idx in set_B:
            if check_stabilizer_2d(bitstring, qubit_idx, grid_size, n_qubits) == -1:
                all_satisfied = False
                break
        if all_satisfied:
            success_count_B += count
    
    prob_B = success_count_B / shots
    
    # --- 6. COMPUTE WITNESS VALUE ---
    w_value = 3 - 2 * (prob_A + prob_B)
    
    return w_value, prob_A, prob_B

def create_noise_model(single_qubit_error=0.001, two_qubit_error=0.01, readout_error=0.02):
    """
    Creates a realistic noise model with gate and measurement errors.
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

def visualize_grid(grid_size):
    """Visualize the checkerboard pattern for the square graph state."""
    set_A, set_B = get_checkerboard_sets(grid_size)
    
    print(f"\n{grid_size}x{grid_size} Grid Layout:")
    print("=" * (grid_size * 4 + 1))
    for row in range(grid_size):
        row_str = ""
        for col in range(grid_size):
            idx = coord_to_index(row, col, grid_size)
            if idx in set_A:
                row_str += f"[A{idx:2d}]"
            else:
                row_str += f"[B{idx:2d}]"
        print(row_str)
    print("=" * (grid_size * 4 + 1))
    print("A = White squares (measure X on these)")
    print("B = Black squares (measure X on these)\n")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("=== Square Graph State Witness ===")
    print("Noise Parameters:")
    print("  Single-qubit gates (H): 0.2% error")
    print("  Two-qubit gates (CZ): 2.0% error")
    print("  Readout: 1.7% error\n")
    
    # Create noisy simulator
    noise_model = create_noise_model(
        single_qubit_error=0.001,
        two_qubit_error=0.02,
        readout_error=0.002
    )
    #backend_sim = AerSimulator(noise_model=noise_model)
    provider = IQMProvider(
        "https://resonance.meetiqm.com",
        quantum_computer="emerald",
        token="kXL7TYp+aF382y0PoH+iJ9bfYPCbhwDt8fZCu7KHoaMBnBezagx+Q5zUHT1QCtkp",
    )
    backend = provider.get_backend()

    # Test different square grid sizes
    for grid_size in [2, 3, 4]:
        visualize_grid(grid_size)
        w, pA, pB = run_square_witness(grid_size, backend, shots=5000)
        print(f"{grid_size}x{grid_size} Grid: Witness = {w:.3f} (P_white={pA:.3f}, P_black={pB:.3f})")
        print(f"  -> Entanglement: {'✓ YES' if w < 0 else '✗ NO'} (W < 0 required)\n")
    
    print("\nExpected: W < 0 indicates genuine multipartite entanglement")
    print("Perfect graph state would give W = -1.0")
