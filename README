# DNA Sequence Alignment Project

## Overview

This project implements and optimizes a **DNA sequence alignment** algorithm using a **brute-force** approach, with parallelization strategies based on **OpenMP**, **MPI**, **MPI+OpenMP**, and **CUDA**. The program searches for **exact matches** of nucleotide patterns within a generated DNA sequence, and evaluates how different parallelization models impact performance.

## 🧬 Problem Description

Given a main DNA sequence of length *N* and a set of *M* patterns (random or sampled), the algorithm finds the **first occurrence** of each pattern in the sequence, and tracks how many patterns align on each position.

The DNA sequence is generated pseudo-randomly with specific nucleotide frequencies (A, C, G, T). The patterns can be random or subsequences taken from the main sequence, and are generated with user-defined statistical parameters (length, location, deviation, etc.).

## 🔧 Input Parameters

The executable takes **14 parameters**:

1. `sequence_length`
2. `prob_G`
3. `prob_C`
4. `prob_A` *(T is 1 - (G + C + A))*
5. `num_random_patterns`
6. `random_pattern_length_mean`
7. `random_pattern_length_dev`
8. `num_sampled_patterns`
9. `sampled_pattern_length_mean`
10. `sampled_pattern_length_dev`
11. `sampled_pattern_location_mean`
12. `sampled_pattern_location_dev`
13. `sample_mixing`: B (before), A (after), M (mixed)
14. `seed`

## 🚀 Implementations

### 🔹 OpenMP

* Parallelizes the main loop using `#pragma omp parallel for reduction(+:pat_matches)`
* Uses `schedule(dynamic,4)` to balance variable workload across threads
* Handles `seq_matches` with `#pragma omp atomic` to avoid race conditions
* Optimized to reduce **false sharing** and memory contention

### 🔸 MPI

* Distributes patterns statically among processes
* Uses `MPI_Reduce`, `MPI_Gather`, `MPI_Gatherv` to aggregate results
* Handles communication asynchronously with `MPI_Isend` / `MPI_Irecv`
* Avoids initialization bugs and ensures correct reduction semantics

### 🔸 MPI + OpenMP (Hybrid)

* MPI handles pattern distribution across processes
* OpenMP accelerates intra-process computation
* Uses dynamic scheduling (`schedule(dynamic,64)`) to balance load
* MPI communication remains serialized to avoid thread-safety issues
* Improved speedup and memory usage compared to pure MPI

### 🟩 CUDA

* Implements brute-force pattern matching on GPU with a 2D grid layout:

  * **X-axis**: sequence positions
  * **Y-axis**: patterns
* Uses `atomicCAS` to track first match, `atomicAdd` for match count
* Optimizes memory access (coalesced reads, avoids shared memory)
* Handles edge-cases: overlapping chunks, GPU-compatible match flag (`ULLONG_MAX`)

## ✅ Output

* Number of patterns matched
* Coverage count for each sequence position
* Checksum values to validate correctness
* Execution time (excluding setup/init)

## 🧪 Performance Results

* CUDA achieved a **90x acceleration** over the sequential version
* MPI+OpenMP offered the best tradeoff between scalability and ease of implementation
* OpenMP alone was the most stable for up to 8 threads

### 📊 Strong/Weak Scaling & Efficiency

* CUDA dominates in strong scaling with large inputs
* OpenMP scales better than MPI for small thread counts
* Hybrid (MPI+OMP) benefits from cache locality and shared memory

## 📁 Project Structure

```
sequenceProject/
├── cuda/         # CUDA implementation
├── mpi/          # MPI-only implementation
├── omp/          # OpenMP-only implementation
├── hybrid/       # MPI + OpenMP
├── sequential/   # Original reference sequential version
├── common/       # Shared code: RNG, data generation, checksum
├── Makefile      # Build targets for all versions
```

## 📦 Build Instructions

Make sure you have a C compiler, MPI and CUDA installed, then:

```bash
make              # builds all versions
make cuda         # builds only the CUDA version
make omp          # builds only OpenMP version
make mpi          # builds MPI version
make hybrid       # builds hybrid version
```

To enable debug mode:

```bash
make DEBUG=1
```

## ▶️ Run Example

```bash
.align_omp 700000 0.2 0.2 0.3 100 8 2 50 10 2 300000 10000 B 123456
```

## 👥 Authors

* Matteo Sperini – [@teosperini](https://github.com/teosperini)
* Giordana Foglia – [@giordanaf](https://github.com/ooojordan)
