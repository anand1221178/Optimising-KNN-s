import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("knn_scaling_results.csv")

# Unique K values
k_vals = sorted(df['k'].unique())

# Plot Runtime vs Threads
plt.figure(figsize=(10, 6))
for k in k_vals:
    subset = df[df['k'] == k]
    plt.plot(subset['num_threads'], subset['runtime'], marker='o', label=f'k={k}')
plt.title("KNN Runtime vs Number of Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Runtime (s)")
plt.grid(True)
plt.legend()
plt.savefig("knn_runtime_vs_threads.png")
plt.show()

# Plot Speedup
plt.figure(figsize=(10, 6))
for k in k_vals:
    subset = df[df['k'] == k]
    plt.plot(subset['num_threads'], subset['speedup'], marker='o', label=f'k={k}')
plt.title("KNN Speedup vs Number of Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup (X)")
plt.grid(True)
plt.legend()
plt.savefig("knn_speedup_vs_threads.png")
plt.show()

# Plot Efficiency
plt.figure(figsize=(10, 6))
for k in k_vals:
    subset = df[df['k'] == k]
    plt.plot(subset['num_threads'], subset['efficiency'], marker='o', label=f'k={k}')
plt.title("KNN Parallel Efficiency vs Number of Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Efficiency (%)")
plt.grid(True)
plt.legend()
plt.savefig("knn_efficiency_vs_threads.png")
plt.show()
