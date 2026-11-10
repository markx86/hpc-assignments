# HPC Assignment #1: Optimization of ADI (Alternating-Direction Implicit)

This project focuses on the parallelization and performance optimization of the **ADI (Alternating-Direction Implicit)** integration kernel from the [Polybench/C](https://sourceforge.net/p/polybench/wiki/Home/) benchmark suite using **OpenMP**.

---

## Authors Information

- **Raffaele Cimino**  
  - Email: [ciminoraffaele04@hotmail.com](mailto:ciminoraffaele04@hotmail.com)  
  - Role: Implementation of optimized OpenMP kernels (`adi_v1_rc.c`, `adi_v2_rc.c`, `adi_v3_rc.c`), algorithmic optimizations (parallel region hoisting, loop fusion, matrix transposition for cache locality), and test script improvements.

- **markx86**  
  - Profile: [markx86](https://github.com/markx86) / `<markx86@noreply.codeberg.org>`  
  - Role: Initial repository structure, baseline reference implementation, Makefile build system, and initial verification framework.

---

## Project Structure & Implementations

- **`adi_ref.c` (Reference - `ref`)**:
  - Pure sequential baseline version directly from Polybench.
- **`adi_v1.c` (Baseline OpenMP - `v1`)**:
  - Naive OpenMP parallelization placing `#pragma omp parallel for` directly on individual inner loops inside the timestep iteration loop. Suffers from high fork-join overhead.
- **`adi_v1_rc.c` (Persistent Parallel Region - `v1_rc`)**:
  - Hoists `#pragma omp parallel` outside the outer timestep (`t`) loop, keeping the thread team alive and using `#pragma omp for` inside. Eliminates thread team recreation overhead at each timestep.
- **`adi_v2_rc.c` (Loop Fusion - `v2_rc`)**:
  - Fuses the three row-wise computation blocks inside a single `#pragma omp for` iteration space over rows (`i1`). Drastically reduces synchronization barriers between row passes.
- **`adi_v3_rc.c` (Task-Based Parallelism & Column Loop Inversion - `v3_rc`)**:
  - Implements an asynchronous, task-based parallelization model using OpenMP `#pragma omp taskloop num_tasks(4 * nthreads)` for dynamic load balancing across threads, orchestrated by a single master thread (`#pragma omp single`) driving the timestep loop.
  - Inverts the loop hierarchy in Phase 2 (column sweep) by hoisting the independent column index `i2` to the outer loop, fusing forward elimination, boundary division, and back-substitution into self-contained column tasks.
  - Slashes OpenMP synchronization barriers from $O(N)$ (~2047) per timestep down to strictly 2 taskloop barriers. Captures the architectural trade-off between barrier elimination and the Stride-N memory traversal overhead.
- **`adi.h`**:
  - Problem definitions, dataset dimension presets (`MINI`, `SMALL`, `STANDARD`, `LARGE`, `EXTRALARGE`), and data types.
- **`common/`**:
  - Polybench support utilities for accurate timing measurement (`POLYBENCH_TIME`) and data dumping (`POLYBENCH_DUMP_ARRAYS`).
- **`Makefile`**:
  - GNU Make build system configured with OpenMP (`-fopenmp`), optimization level `-O2`, and dataset macros.
- **`test.sh`**:
  - Automated bash verification script that compares output matrices against the reference sequential run.

---

## Building and Configuration

The build system is driven by `make`.

### Available Targets
- `make build`: Compiles the binary without executing it.
- `make run`: Compiles (if needed) and executes the target binary.
- `make clean`: Removes all compiled `adi_*` executables.

### Build Variables
You can pass the following variables to `make`:
- **`VER`**: Specifies which implementation to build (default: `ref`).  
  Available values: `ref`, `v1`, `v1_rc`, `v2_rc`, `v3_rc`.
- **`DATASET`**: Specifies the problem dimensions (default: `mini`).  
  Available sizes:
  - `mini`: $N = 32$, $TSTEPS = 2$
  - `small`: $N = 500$, $TSTEPS = 10$
  - `standard`: $N = 1024$, $TSTEPS = 50$ (Polybench default)
  - `large`: $N = 2000$, $TSTEPS = 50$
  - `extralarge`: $N = 4000$, $TSTEPS = 100$
- **`NTHREADS`**: Sets the compilation macro for thread count (default: host core count via `nproc`).
- **`DUMP_ARRAYS`**: When set (e.g. `DUMP_ARRAYS=1`), enables output array dumping to `stderr` for numerical verification.

---

## Commands for Reproducing the Experiments

All execution times are measured directly by the Polybench timing harness in seconds.

### 1. Clean Build Environment
Always ensure previous builds are cleared before running benchmarks:
```sh
make clean
```

### 2. Baseline Sequential Benchmark
Measure the execution time of the sequential reference implementation:
```sh
# Run with standard dataset
make run VER=ref DATASET=standard

# Run with large dataset
make run VER=ref DATASET=large
```

### 3. Benchmarking Parallel Versions with Default Threads
Build and run each parallel version:
```sh
# Naive OpenMP baseline
make run VER=v1 DATASET=standard

# Persistent Parallel Region (Raffaele v1)
make run VER=v1_rc DATASET=standard

# Row-wise Loop Fusion (Raffaele v2)
make run VER=v2_rc DATASET=standard

# OpenMP Taskloop & Column Inversion (Raffaele v3)
make run VER=v3_rc DATASET=standard
```

### 4. Scalability Experiments (Varying Thread Counts)
To test strong scaling, vary the number of OpenMP threads using both the compile-time `NTHREADS` flag and the runtime environment variable `OMP_NUM_THREADS`:

```sh
# Example: Running v2_rc across 1, 2, 4, 8, 16 threads on standard dataset
for t in 1 2 4 8 16; do
    echo "=== Testing VER=v2_rc with $t threads ==="
    make clean > /dev/null
    make build VER=v2_rc DATASET=standard NTHREADS=$t > /dev/null
    OMP_NUM_THREADS=$t ./adi_v2_rc
done
```

Repeat the above loop for `v1`, `v1_rc`, and `v3_rc` to record speedup and efficiency curves.

### 5. Dataset Scaling Experiments
Evaluate how the task-based and column-inverted version (`v3_rc`) behaves across varying matrix dimensions compared to `v2_rc` and the sequential reference:
```sh
for ds in small standard large extralarge; do
    echo "=== Dataset: $ds ==="
    echo -n "Reference (seq): "
    make run VER=ref DATASET=$ds 2>/dev/null
    
    echo -n "Loop Fusion (v2_rc): "
    make run VER=v2_rc DATASET=$ds 2>/dev/null
    
    echo -n "Taskloop & Inversion (v3_rc): "
    make run VER=v3_rc DATASET=$ds 2>/dev/null
done
```

### 6. Full Benchmark Suite Automation Script
You can run this shell one-liner to benchmark all versions across thread counts and output a formatted comparison table:
```sh
DATASET=standard
echo "Version,Threads,Time(s)" > benchmark_results.csv
for ver in v1 v1_rc v2_rc v3_rc; do
    for t in 1 2 4 8 16; do
        make clean > /dev/null
        make build VER=$ver DATASET=$DATASET NTHREADS=$t > /dev/null
        exec_time=$(OMP_NUM_THREADS=$t ./adi_$ver 2>/dev/null)
        echo "$ver,$t,$exec_time" >> benchmark_results.csv
        echo "$ver | Threads: $t | Time: $exec_time s"
    done
done
```

---

## Verification & Correctness Checking

To verify that the parallel optimizations produce identical results to the reference code, use the verification script `test.sh`.

### Running the Test Script
```sh
# Test a specific version with standard dataset
./test.sh v1_rc
./test.sh v2_rc
./test.sh v3_rc

# Test with a specific dataset size (e.g. mini or small)
DATASET=small ./test.sh v2_rc
DATASET=mini ./test.sh v3_rc
```

### How Verification Works
1. The script compiles `adi_ref` with `DUMP_ARRAYS=1`, dumping the matrix elements formatted to standard error (`stderr`).
2. The reference output is cached in `/tmp/hpc-lab-ref_$DATASET.txt`.
3. The specified parallel version is compiled with `DUMP_ARRAYS=1` and its output redirected to a temporary file via `mktemp`.
4. A file comparison (`diff -q`) checks that all elements match the sequential reference.
