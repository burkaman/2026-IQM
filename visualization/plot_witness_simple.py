import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

# Data from the image
qubits = [6, 14, 16, 19, 20, 24]
witness_values = [-1, -0.795, -0.709, -0.377, -0.719, -0.75]

# Set font
available_fonts = [f.name for f in fm.fontManager.ttflist]
if "Red Hat Display" in available_fonts:
    plt.rcParams["font.family"] = "Red Hat Display"
else:
    plt.rcParams["font.family"] = "sans-serif"

# Create the plot
plt.figure(figsize=(12, 7))

# Plot single line without error bars
plt.plot(
    qubits,
    witness_values,
    marker="o",
    linestyle="-",
    linewidth=2,
    color="#2E86AB",
    markersize=6,
    label="Optimized",
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
plt.ylabel("Witness Value (W)", fontsize=14, fontweight="bold")
plt.title("Garnet Witness Values", fontsize=15, fontweight="bold", pad=20)
plt.legend(fontsize=12, loc="upper right")
plt.grid(True, alpha=0.3, linestyle=":", linewidth=1)

# Set x-axis to show the qubit counts in the data
plt.xticks(qubits)
plt.xlim(min(qubits) - 1, max(qubits) + 1)

# Set y-axis maximum
plt.ylim(-1, 1)

# Add shaded region for GME (W < 0)
plt.axhspan(-1, 0, alpha=0.1, color="green", label="GME Region")

plt.tight_layout()
plt.savefig("witness_plot_simple.png", dpi=300, bbox_inches="tight")
print("Plot saved to witness_plot_simple.png")
plt.show()
