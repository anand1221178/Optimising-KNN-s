# Parallel K-Nearest Neighbors (KNN) Classification with OpenMP

This project implements a K-Nearest Neighbors (KNN) classifier in C++ with three execution models:

- ✅ **Serial**: Baseline reference without any parallelization.
- 🚀 **Parallel (Task-based)**: Uses OpenMP `task` directives in QuickSort.
- ⚡ **Parallel (Sections-based)**: Uses OpenMP `sections` to parallelize QuickSort.

It also generates performance metrics (runtime, speedup, efficiency, accuracy) across different thread counts and values of `k`.

---

## 🧠 KNN Overview

KNN is a simple machine learning algorithm used for classification. For each test sample:
1. Compute the Euclidean distance to every training sample.
2. Sort the distances.
3. Pick the majority label among the `k` nearest neighbors.

---

## 🛠️ Requirements

- A C++17 compatible compiler (e.g., `g++`)
- OpenMP support enabled (usually `-fopenmp`)
- Training and testing data in binary `.bin` format:
  - Features: Flattened float32 arrays of shape `(num_samples, feature_dim)`
  - Labels: int32 array of labels

---

## 📁 Directory Structure

```
.
├── knn.cpp                  # Main program file
├── run_knn.sh              # SLURM batch script for cluster execution
├── train/
│   ├── train_features.bin
│   └── train_labels.bin
├── test/
│   ├── test_features.bin
│   └── test_labels.bin
```

---

## 🧪 Running the Code

### 🧰 Compile

```bash
make
```

### 🚀 Execute Locally

```bash
./knn
```

### 🧠 Execute on Cluster

```bash
sbatch run_knn.sh
```

---

## 📊 Outputs

1. **Console**: Displays per-thread timings, speedups, and accuracies.
2. **CSV Files**:
   - `knn_scaling_results.csv`: Parallel KNN using task-based QuickSort.
   - `knn_sections_vs_tasks.csv`: Comparison between task-based and section-based parallel sorting.

Each CSV file contains:

```csv
k,num_threads,method,runtime,speedup,efficiency,accuracy
```

---

## 🧪 Example

With `k = 3, 5, 7`, the program benchmarks each configuration for:
- Serial
- Parallel Task-Based (`method = task`)
- Parallel Sections-Based (`method = sections`)

---

## 📌 Notes

- Sorting uses QuickSort, which is parallelized via either `omp task` or `omp section`.
- The feature vector is of dimension 512.
- Default: 50,000 training samples and 10,000 test samples.
- `MAX_TASK_DEPTH = 3` limits over-parallelization in recursion.

---

## 🧼 Cleaning Up

```bash
make clean
```

---

## 👨‍💻 Author

Anand Patel