# HPC Assignment #1: Optimization of ADI (Alternating-Direction Implicit)

This project focuses on the parallelization and performance optimization of the **ADI (Alternating-Direction Implicit)** integration kernel from the [Polybench/C](https://sourceforge.net/p/polybench/wiki/Home/) benchmark suite using **OpenMP**.

---

- **Raffaele Cimino**  
  - Email: [401858@studenti.unimore.it](mailto:401858@studenti.unimore.it)  
  - Role: Implementation of optimized OpenMP kernels (`adi_v1_rc.c`, `adi_v2_rc.c`, `adi_v3_rc.c`), algorithmic optimizations (parallel region hoisting, loop fusion, matrix transposition for cache locality), and test script improvements.
  - **Sorelle Sandra Simo Defo**
  - Email: [@studenti.unimore.it](mailto:@studenti.unimore.it)   

## Authors Information

- **Raffaele Cimino**  
  - GitHub: [@Rafxf04](https://github.com/Rafxf04) 
  - Email: [401858@studenti.unimore.it](mailto:401858@studenti.unimore.it)   
  - Role: Implementation of optimized OpenMP kernels (`adi_v1_rc.c`, `adi_v2_rc.c`, `adi_v3_rc.c`), algorithmic optimizations (persistent parallel region / fork-join elimination, row-wise loop fusion, OpenMP taskloop with column loop inversion), benchmark presentation (`Problem-Anatomy-and-Profiling-the-ADI-Alternating-Direction-Implicit-Stencil-Solver.pptx`), and test script improvements.

- **Marco Paini**  
  - GitHub: [@markx86](https://github.com/markx86) 
  - Email: [402991@studenti.unimore.it](mailto:402991@studenti.unimore.it)    
  - Role: Initial repository setup, baseline Polybench integration, Makefile build system, initial verification framework (`test.sh`), and OpenMP implementations (`adi_v1_mp.c`, `adi_v2_mp.c`, `adi_v3_mp.c`, `adi_v4_mp.c` focusing on fork-join overhead, cache-line-aligned chunking, and barrier optimizations).

- **Sorelle Sandra Simo Defo** (Sandra0502)  
  - GitHub: [@Sandra0502](https://github.com/Sandra0502)  
  - Email: [403213@studenti.unimore.it](mailto:403213@studenti.unimore.it)  
  - Role: Implementation of OpenMP parallelization variants (`adi_metodo1.c` and `adi_metodo2.c`).

---

## Project Structure & Implementations

### Reference Implementation
- **`adi_ref.c` (Reference - `ref`)**:
  - Pure sequential baseline version directly from Polybench/C.

### Raffaele Cimino Implementations (`_rc`)
- **`adi_v1_rc.c` (Persistent Parallel Region - `v1_rc`)**:
  - Hoists `#pragma omp parallel` outside the outer timestep (`t`) loop, keeping the thread team alive and using `#pragma omp for` inside. Eliminates thread team recreation overhead at each timestep.
- **`adi_v2_rc.c` (Loop Fusion - `v2_rc`)**:
  - Fuses the three row-wise computation blocks inside a single `#pragma omp for` iteration space over rows (`i1`). Drastically reduces synchronization barriers between row passes and maximizes L1/L2 cache residency.
- **`adi_v3_rc.c` (Task-Based Parallelism & Column Loop Inversion - `v3_rc`)**:
  - Implements an asynchronous, task-based parallelization model using OpenMP `#pragma omp taskloop num_tasks(4 * nthreads)` for dynamic load balancing across threads, orchestrated by a single master thread (`#pragma omp single`) driving the timestep loop.
  - Inverts the loop hierarchy in Phase 2 (column sweep) by hoisting the independent column index `i2` to the outer loop, fusing forward elimination, boundary division, and back-substitution into self-contained column tasks.
  - Slashes OpenMP synchronization barriers from $O(N)$ (~2047) per timestep down to strictly 2 taskloop barriers. Captures the architectural trade-off between barrier elimination and the Stride-N memory traversal overhead.

### markx86 Implementations (`_mp`)
- **`adi_v1_mp.c` (Baseline OpenMP - `v1_mp`)**:
  - Naive OpenMP parallelization placing `#pragma omp parallel for` directly on individual inner loops inside the timestep iteration loop. Suffers from high fork-join overhead.
- **`adi_v2_mp.c` (Row-wise Fused OpenMP - `v2_mp`)**:
  - Merges row operations in Phase 1 into a single `#pragma omp parallel for` block while maintaining per-step thread dispatch, evaluating the impact of fork-join synchronization.
- **`adi_v3_mp.c` (Persistent Parallel Region - `v3_mp`)**:
  - Hoists `#pragma omp parallel` across the timestep loop, executing row and column passes through `#pragma omp for` worksharing directives.
- **`adi_v4_mp.c` (Cache-Aligned Chunking & Barrier Tuning - `v4_mp`)**:
  - Implements custom work distribution using cache-line-aligned chunk sizes (`ELEMS_PER_CLINE`), explicit per-thread column bounds (`th_i2_start`, `th_i2_end`), and fine-tuned barrier synchronization.

### Sandra Implementations (`metodo*`)
- **`adi_metodo1.c` (OpenMP Method 1 - `metodo1`)**:
  - Parallelization using multiple `#pragma omp parallel for` directives on each inner loop within the timestep iteration.
- **`adi_metodo2.c` (OpenMP Method 2 - `metodo2`)**:
  - Parallelization using a single `#pragma omp parallel` region per timestep with `#pragma omp for` worksharing directives for each phase.

### Headers, Build System & Documentation
- **`adi.h`**:
  - Problem definitions, dataset dimension presets (`MINI`, `SMALL`, `STANDARD`, `LARGE`, `EXTRALARGE`), and data types (`DATA_TYPE`).
- **`common/`**:
  - Polybench support utilities for accurate timing measurement (`POLYBENCH_TIME`) and array dumping (`POLYBENCH_DUMP_ARRAYS`):
    - `common/polybench.c`: Core timing and array print routines.
    - `common/polybench.h`: Definitions, configuration switches, and macro helpers.
- **`Makefile`**:
  - GNU Make build system configured with OpenMP (`-fopenmp`), optimization level `-O2`, and dataset macros. Supports compiling single targets or `all`.
- **`test.sh`**:
  - Automated bash verification script that compares output matrices against the reference sequential run (`adi_ref`).
- **`Problem-Anatomy-and-Profiling-the-ADI-Alternating-Direction-Implicit-Stencil-Solver.pptx`**:
  - Presentation and profiling analysis of the ADI stencil solver on an NVIDIA Jetson Nano (quad-core ARM Cortex-A57), covering performance metrics, cache locality, SIMD vectorization, and efficiency comparisons.

---

## Building and Configuration

The build system is driven by `make`.

### Available Targets
- `make build`: Compiles the binary without executing it.
- `make run`: Compiles (if needed) and executes the target binary.
- `make all`: Compiles all available `adi_*` implementations.
- `make clean`: Removes all compiled `adi_*` executables.

### Build Variables
You can pass the following variables to `make`:
- **`VER`**: Specifies which implementation to build (default: `ref`).  
  Available values:
  - **Sequential Baseline**: `ref`
  - **Raffaele Cimino Implementations**: `v1_rc`, `v2_rc`, `v3_rc`
  - **markx86 Implementations**: `v1_mp`, `v2_mp`, `v3_mp`, `v4_mp`
  - **Sandra Implementations**: `metodo1`, `metodo2`
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
# Raffaele Cimino implementations
make run VER=v1_rc DATASET=standard
make run VER=v2_rc DATASET=standard
make run VER=v3_rc DATASET=standard

# markx86 implementations
make run VER=v1_mp DATASET=standard
make run VER=v2_mp DATASET=standard
make run VER=v3_mp DATASET=standard
make run VER=v4_mp DATASET=standard

# Sandra implementations
make run VER=metodo1 DATASET=standard
make run VER=metodo2 DATASET=standard
```

## Verification & Correctness Checking

To verify that the parallel optimizations produce identical results to the reference code, use the verification script `test.sh`.

### Running the Test Script
```sh
# Test specific versions with standard dataset
./test.sh v1_rc
./test.sh v2_rc
./test.sh v3_rc
./test.sh v1_mp
./test.sh v2_mp
./test.sh v3_mp
./test.sh v4_mp
./test.sh metodo1
./test.sh metodo2

# Test with a specific dataset size (e.g. mini or small)
DATASET=small ./test.sh v2_rc
DATASET=mini ./test.sh v3_rc
```

### How Verification Works
1. The script (`./test.sh`) compiles `adi_ref` with `DUMP_ARRAYS=1`, dumping the matrix elements formatted to standard error (`stderr`).
2. The reference output is cached in `/tmp/hpc-lab-ref_$DATASET.txt`.
3. The specified parallel version is compiled with `DUMP_ARRAYS=1` and its output redirected to a temporary file via `mktemp`.
4. A file comparison (`diff -q`) checks that all elements match the sequential reference.
