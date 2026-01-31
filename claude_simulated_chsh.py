import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator


# Create a circuit to prepare the Bell state |ψ⟩ = (|00⟩ + |11⟩)/√2
def create_bell_state():
    """Create a Bell state (maximally entangled state)"""
    qc = QuantumCircuit(2, 2)
    qc.h(0)  # Hadamard on first qubit
    qc.cx(0, 1)  # CNOT to create entanglement
    return qc


# Measurement circuits for different bases
def measure_in_basis(qc, qubit, angle):
    """
    Rotate measurement basis by angle (in degrees) before measuring
    This effectively measures in a rotated basis
    """
    qc.ry(-2 * np.radians(angle), qubit)  # Rotate basis
    qc.measure(qubit, qubit)
    return qc


# Create circuits for the four measurement combinations
def create_chsh_circuits():
    """
    Create circuits for CHSH inequality test
    Alice measures in Z basis (A1, angle=0) or X basis (A2, angle=90)
    Bob measures at 45° (B1, angle=45) or 135° (B2, angle=135)
    """
    circuits = {}
    measurement_settings = {
        "A1B1": (0, 45),  # Z ⊗ (Z+X)/√2
        "A1B2": (0, -45),  # Z ⊗ (Z-X)/√2
        "A2B1": (90, 45),  # X ⊗ (Z+X)/√2
        "A2B2": (90, -45),  # X ⊗ (Z-X)/√2
    }

    for name, (alice_angle, bob_angle) in measurement_settings.items():
        qc = create_bell_state()
        qc.barrier()

        # Apply rotation for Alice's measurement basis
        qc.ry(-2 * np.radians(alice_angle), 0)
        # Apply rotation for Bob's measurement basis
        qc.ry(-2 * np.radians(bob_angle), 1)

        qc.barrier()
        qc.measure([0, 1], [0, 1])

        circuits[name] = qc

    return circuits


# Calculate correlation E(a,b) = P(same) - P(different)
def calculate_correlation(counts, shots):
    """Calculate correlation E = P(00) + P(11) - P(01) - P(10)"""
    same = counts.get("00", 0) + counts.get("11", 0)
    diff = counts.get("01", 0) + counts.get("10", 0)
    return (same - diff) / shots


# Main execution
def demonstrate_chsh_violation():
    """Run the full CHSH experiment"""
    print("=" * 70)
    print("CHSH Inequality Violation Demonstration")
    print("=" * 70)

    # Create circuits
    circuits = create_chsh_circuits()

    # Display one circuit as example
    print("\nExample circuit (A1B1 measurement):")
    print(circuits["A1B1"])

    # Run simulation
    simulator = AerSimulator()
    shots = 8192

    print(f"\nRunning simulation with {shots} shots...")

    # Execute all circuits
    results = {}
    correlations = {}

    for name, circuit in circuits.items():
        job = simulator.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        results[name] = counts
        correlations[name] = calculate_correlation(counts, shots)

    # Display results
    print("\n" + "=" * 70)
    print("MEASUREMENT RESULTS")
    print("=" * 70)

    for name in ["A1B1", "A1B2", "A2B1", "A2B2"]:
        print(f"\n{name}: E = {correlations[name]:.4f}")
        print(f"  Counts: {results[name]}")

    # Calculate CHSH parameter S
    S = (
        correlations["A1B1"]
        - correlations["A1B2"]
        + correlations["A2B1"]
        + correlations["A2B2"]
    )

    print("\n" + "=" * 70)
    print("CHSH INEQUALITY TEST")
    print("=" * 70)
    print(f"\nS = E(A₁,B₁) - E(A₁,B₂) + E(A₂,B₁) + E(A₂,B₂)")
    print(
        f"S = {correlations['A1B1']:.4f} - {correlations['A1B2']:.4f} + {correlations['A2B1']:.4f} + {correlations['A2B2']:.4f}"
    )
    print(f"\nS = {S:.4f}")
    print(f"\nClassical bound: |S| ≤ 2")
    print(f"Quantum prediction: S = 2√2 ≈ 2.828")
    print(f"Measured value: S = {S:.4f}")

    if abs(S) > 2:
        print(f"\n✓ CHSH inequality VIOLATED! ({abs(S):.4f} > 2)")
        print("This demonstrates quantum entanglement!")
    else:
        print(f"\n✗ No violation detected (might need more shots)")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "CHSH Inequality: Measurement Outcomes", fontsize=16, fontweight="bold"
    )

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    names = ["A1B1", "A1B2", "A2B1", "A2B2"]

    for idx, (pos, name) in enumerate(zip(positions, names)):
        ax = axes[pos]
        counts = results[name]
        ax.bar(
            counts.keys(),
            counts.values(),
            color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"][: len(counts)],
        )
        ax.set_title(f"{name}: E = {correlations[name]:.4f}", fontweight="bold")
        ax.set_xlabel("Measurement Outcome")
        ax.set_ylabel("Counts")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/claude/chsh_results.png", dpi=300, bbox_inches="tight")
    print("\n✓ Results visualization saved to 'chsh_results.png'")

    # Create S value comparison chart
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

    categories = ["Classical\nBound", "Quantum\nPrediction", "Measured\nValue"]
    values = [2, 2 * np.sqrt(2), abs(S)]
    colors = ["#95a5a6", "#9b59b6", "#e74c3c"]

    bars = ax2.bar(
        categories, values, color=colors, alpha=0.8, edgecolor="black", linewidth=2
    )

    # Add horizontal line at classical bound
    ax2.axhline(
        y=2, color="red", linestyle="--", linewidth=2, label="Classical Limit |S| ≤ 2"
    )

    # Annotate bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    ax2.set_ylabel("|S| Value", fontsize=12, fontweight="bold")
    ax2.set_title("CHSH Parameter Comparison", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 3)

    plt.tight_layout()
    plt.savefig("/home/claude/chsh_comparison.png", dpi=300, bbox_inches="tight")
    print("✓ S value comparison saved to 'chsh_comparison.png'")

    return circuits, results, S


# Run the demonstration
circuits, results, S_value = demonstrate_chsh_violation()
