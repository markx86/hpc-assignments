/* POLYBENCH/GPU-OPENMP
 *
 * This file is a part of the Polybench/GPU-OpenMP suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 *
 * Copyright 2013, The University of Delaware
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <omp.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 10x1024x1024. */
#include "adi.h"


#define GS 32 //branching factor numero di operazioni per singolo thread del task //grain size

/* Array initialization. */
static void init_array(int n,
                       DATA_TYPE POLYBENCH_2D(X, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(B, N, N, n, n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      X[i][j] = ((DATA_TYPE)i * (j + 1) + 1) / n;
      A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / n;
      B[i][j] = ((DATA_TYPE)i * (j + 3) + 3) / n;
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        DATA_TYPE POLYBENCH_2D(X, N, N, n, n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, X[i][j]);
      if ((i * N + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* ============================================================================
 * KERNEL: ADI - Version 3_rc (Task-Based Parallelism & Column Loop Inversion)
 * Author: Raffaele Cimino
 *
 * Architectural & Algorithmic Strategy:
 *
 *   1. Task-Based Parallelism via OpenMP Taskloop:
 *      - Master-Worker Coordination: A single parallel region (`#pragma omp parallel`)
 *        creates the persistent thread team. A single thread (`#pragma omp single`)
 *        drives the outer sequential timestep loop `t`.
 *      - Task Granularity & Dynamic Balancing: Iterations of both the row and
 *        column sweeps are partitioned into explicit tasks using
 *        `#pragma omp taskloop num_tasks(4 * nthreads)`.
 *        Oversubscribing by a factor of 4 tasks per thread provides dynamic
 *        load balancing, absorbing core execution variance and thread preemption.
 *
 *   2. Algorithmic Inversion & Full Fusion in Phase 2 (Column Sweep):
 *      - Inversion of Loops: In baseline Polybench and v2_rc, the vertical recurrence
 *        along `i1` was located in the outer loop, necessitating fine-grained worksharing
 *        on the inner loop `i2` and generating $(2N - 1)$ synchronization barriers
 *        per timestep.
 *      - Independence Insight: Each vertical column `i2` represents a completely
 *        independent tridiagonal linear system (Thomas algorithm).
 *      - By hoisting `for (i2 = 0; i2 < N; i2++)` to the outside, all three
 *        sub-steps (forward elimination, boundary division, backward substitution)
 *        are fused sequentially per column task.
 *      - Barrier Elimination: Reduces global synchronization barriers per timestep
 *        from $O(N)$ down to strictly 2 (implicit taskwait boundaries at the
 *        end of each taskloop construct).
 *
 *   3. Architectural Trade-off Considerations:
 *      - Hoisting `i2` to the outside causes the inner loop on `i1` to traverse
 *        memory with a stride of `N * sizeof(double)` (Row-Major layout).
 *      - Task creation, queue dispatch, and task-stealing introduce runtime
 *        scheduling overhead at each timestep compared to static loop partitioning.
 * ============================================================================ */
static void kernel_adi(int tsteps,
                       int n,
                       DATA_TYPE POLYBENCH_2D(X, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(B, N, N, n, n))
{
  int t, i1, i2;

  /* Configure task oversubscription: 4 tasks per available thread */
  int nthreads = omp_get_max_threads();
  int total_tasks = 4 * nthreads;

  /* Persistent thread pool active across all timesteps */
  #pragma omp parallel
  #pragma omp single
  {
    /* Single thread orchestrates sequential timestep progression */
    for (t = 0; t < _PB_TSTEPS; t++)
    {
      /* ----------------------------------------------------------------------
       * Phase 1: Row-wise Tridiagonal System Solve
       * Dynamic task-based execution across independent row blocks `i1`.
       * ---------------------------------------------------------------------- */
      #pragma omp taskloop num_tasks(total_tasks) private(i2)
      for (i1 = 0; i1 < _PB_N; i1++)
      {
        /* Step 1.1: Forward elimination along row i1 */
        for (i2 = 1; i2 < _PB_N; i2++)
        {
          X[i1][i2] = X[i1][i2] - X[i1][i2 - 1] * A[i1][i2] / B[i1][i2 - 1];
          B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2 - 1];
        }

        /* Step 1.2: Boundary normalization on rightmost element */
        X[i1][_PB_N - 1] = X[i1][_PB_N - 1] / B[i1][_PB_N - 1];

        /* Step 1.3: Backward substitution along row i1 */
        for (i2 = 0; i2 < _PB_N - 2; i2++)
        {
          X[i1][_PB_N - i2 - 2] =
              (X[i1][_PB_N - 2 - i2] -
               X[i1][_PB_N - 2 - i2 - 1] * A[i1][_PB_N - i2 - 3]) /
              B[i1][_PB_N - 3 - i2];
        }
      }

      /* ----------------------------------------------------------------------
       * Phase 2: Column-wise Tridiagonal System Solve
       * Inverted loop structure: outer loop on `i2` allows complete fusion
       * of all three column sub-steps into dynamic column tasks.
       * ---------------------------------------------------------------------- */
      #pragma omp taskloop num_tasks(total_tasks) private(i1)
      for (i2 = 0; i2 < _PB_N; i2++)
      {
        /* Step 2.1: Forward elimination along column i2 */
        for (i1 = 1; i1 < _PB_N; i1++)
        {
          X[i1][i2] = X[i1][i2] - X[i1 - 1][i2] * A[i1][i2] / B[i1 - 1][i2];
          B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1 - 1][i2];
        }

        /* Step 2.2: Boundary normalization along column i2 */
        X[_PB_N - 1][i2] = X[_PB_N - 1][i2] / B[_PB_N - 1][i2];

        /* Step 2.3: Backward substitution along column i2 */
        for (i1 = 0; i1 < _PB_N - 2; i1++)
        {
          X[_PB_N - 2 - i1][i2] =
              (X[_PB_N - 2 - i1][i2] -
               X[_PB_N - i1 - 3][i2] * A[_PB_N - 3 - i1][i2]) /
              B[_PB_N - 2 - i1][i2];
        }
      }
    }
  }
}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(X, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_adi(tsteps, n, POLYBENCH_ARRAY(X),
             POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(X)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(X);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
