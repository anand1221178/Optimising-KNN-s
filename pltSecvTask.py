
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("knn_sections_vs_tasks.csv")

# Plot speedup comparison
for k in df['k'].unique():
    plt.figure(figsize=(10, 6))
    subset = df[df['k'] == k]
    
    for method in ['task', 'sections']:
        method_data = subset[subset['method'] == method]
        plt.plot(method_data['num_threads'], method_data['speedup'], marker='o', label=method.capitalize())
    
    plt.title(f"KNN Speedup Comparison for k = {k}")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup over Serial")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"knn_speedup_k{k}.png")
    plt.show()

# Plot efficiency comparison
for k in df['k'].unique():
    plt.figure(figsize=(10, 6))
    subset = df[df['k'] == k]
    
    for method in ['task', 'sections']:
        method_data = subset[subset['method'] == method]
        plt.plot(method_data['num_threads'], method_data['efficiency'], marker='s', label=method.capitalize())
    
    plt.title(f"KNN Parallel Efficiency for k = {k}")
    plt.xlabel("Number of Threads")
    plt.ylabel("Efficiency (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"knn_efficiency_k{k}.png")
    plt.show()
