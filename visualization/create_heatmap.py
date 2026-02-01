import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read the data
df = pd.read_csv("heatmap_data.csv", index_col="N")

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    df,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn_r",
    center=0,
    cbar_kws={"label": "Witness"},
    linewidths=0.5,
)

plt.title("Garnet Witness Performance", fontsize=14, pad=20)
plt.xlabel("Circuit Depth", fontsize=12)
plt.ylabel("Qubits", fontsize=12)
plt.tight_layout()

# Save the figure
plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")
print("Heatmap saved as 'heatmap.png'")

# Also display it
plt.show()
