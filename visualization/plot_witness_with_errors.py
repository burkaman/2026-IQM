import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_witness_errors(prob_A, prob_B, shots):
    """
    Calculate error bars for witness values from existing probability data.

    Args:
        prob_A: Measured probability for even-indexed stabilizers
        prob_B: Measured probability for odd-indexed stabilizers
        shots: Number of shots used in the measurement

    Returns:
        sigma_w, sigma_prob_A, sigma_prob_B: Standard errors
    """
    # Binomial standard errors for probabilities
    sigma_prob_A = np.sqrt(prob_A * (1 - prob_A) / shots)
    sigma_prob_B = np.sqrt(prob_B * (1 - prob_B) / shots)

    # Error propagation for W = 3 - 2(P_A + P_B)
    sigma_w = 2 * np.sqrt(sigma_prob_A**2 + sigma_prob_B**2)

    return sigma_w, sigma_prob_A, sigma_prob_B


# Data from the file
optimized_data = [
    (2, -0.927, 0.98, 0.98),
    (3, -0.874, 0.96, 0.97),
    (4, -0.795, 0.96, 0.94),
    (5, -0.758, 0.95, 0.93),
    (6, -0.629, 0.90, 0.91),
    (7, -0.577, 0.93, 0.86),
    (8, -0.486, 0.90, 0.84),
    (9, -0.473, 0.83, 0.91),
    (10, -0.431, 0.84, 0.88),
    (11, -0.334, 0.84, 0.82),
    (12, -0.237, 0.84, 0.78),
    (13, -0.109, 0.78, 0.77),
    (14, -0.026, 0.77, 0.74),
    (15, -0.062, 0.79, 0.74),
    (16, -0.037, 0.77, 0.73),
    (17, 0.243, 0.73, 0.67),
    (18, 0.736, 0.47, 0.67),
]

unoptimized_data = [
    (2, -0.915, 0.98, 0.98),
    (3, -0.711, 0.91, 0.94),
    (4, -0.539, 0.87, 0.90),
    (5, 0.743, 0.31, 0.82),
    (6, -0.365, 0.84, 0.84),
    (7, -0.244, 0.80, 0.82),
    (8, -0.189, 0.78, 0.82),
    (9, -0.106, 0.75, 0.80),
    (10, -0.087, 0.76, 0.78),
    (11, -0.017, 0.73, 0.78),
    (12, 0.022, 0.73, 0.76),
    (13, 0.135, 0.68, 0.75),
    (14, 0.172, 0.68, 0.74),
    (15, 1.071, 0.26, 0.70),
    (16, 0.207, 0.69, 0.71),
    (17, 1.205, 0.26, 0.64),
    (18, 1.073, 0.36, 0.60),
]

# Assuming 2000 shots per measurement (adjust if different)
shots = 2000

# Calculate error bars for optimized data
opt_qubits = [d[0] for d in optimized_data]
opt_witness = [d[1] for d in optimized_data]
opt_errors = [calculate_witness_errors(d[2], d[3], shots)[0] for d in optimized_data]

# Calculate error bars for unoptimized data
unopt_qubits = [d[0] for d in unoptimized_data]
unopt_witness = [d[1] for d in unoptimized_data]
unopt_errors = [
    calculate_witness_errors(d[2], d[3], shots)[0] for d in unoptimized_data
]

available_fonts = [f.name for f in fm.fontManager.ttflist]
if "Red Hat Display" in available_fonts:
    plt.rcParams["font.family"] = "Red Hat Display"
else:
    plt.rcParams["font.family"] = "sans-serif"

# Create the plot
plt.figure(figsize=(12, 7))

# Plot with error bars
plt.errorbar(
    opt_qubits,
    opt_witness,
    yerr=opt_errors,
    marker="o",
    linestyle="-",
    linewidth=2,
    capsize=5,
    label="Optimized",
    color="#2E86AB",
    markersize=6,
)
plt.errorbar(
    unopt_qubits,
    unopt_witness,
    yerr=unopt_errors,
    marker="s",
    linestyle="-",
    linewidth=2,
    capsize=5,
    label="Unoptimized",
    color="#A23B72",
    markersize=6,
)

# Add horizontal line at W=0 (threshold for entanglement)
plt.axhline(
    y=0,
    color="black",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="W=0 (GME threshold)",
)

# Labels and title
plt.xlabel("Qubits", fontsize=14, fontweight="bold")
# plt.ylabel("Witness Value (W)", fontsize=14, fontweight="bold")
plt.title("Garnet Runs", fontsize=15, fontweight="bold", pad=20)
plt.legend(fontsize=12, loc="upper right")
plt.grid(True, alpha=0.3, linestyle=":", linewidth=1)

# Set x-axis to show all qubit counts
plt.xticks(range(2, 19))
plt.xlim(1.5, 18.5)

# Add shaded region for GME (W < 0)
plt.axhspan(-1, 0, alpha=0.1, color="green", label="GME Region")

plt.tight_layout()
plt.savefig("witness_plot_with_errors.png", dpi=300, bbox_inches="tight")
print("Plot saved to witness_plot_with_errors.png")
plt.show()
