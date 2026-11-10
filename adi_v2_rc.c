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

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 10x1024x1024. */
#include "adi.h"

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


/* ============================================================================
 * KERNEL: ADI - Version 2_rc (Row-wise Loop Fusion & Barrier Elimination)
 * Author: Raffaele Cimino
 *
 * Optimization Strategy:
 *   1. Loop Fusion across Row Operations:
 *      In standard Polybench (and adi_v1_rc), the row-wise sweep was partitioned
 *      into 3 separate parallel loops:
 *        - Forward elimination over all rows
 *        - Boundary division over all rows
 *        - Backward substitution over all rows
 *      Crucial Dependency Insight:
 *        For any given row `i1`, the computation is strictly self-contained.
 *        Row `i1` has NO data dependencies on row `i1 - 1` or `i1 + 1` during
 *        Phase 1.
 *      Therefore, all 3 steps for row `i1` can be fused sequentially within
 *      a SINGLE `#pragma omp for` loop over `i1`.
 *
 *   2. Synchronization & Barrier Elimination:
 *      - Fusing these loops eliminates two global OpenMP barrier synchronizations
 *        per timestep.
 *      - Threads process an assigned chunk of rows from start to finish without
 *        pausing to synchronize at intermediate sub-steps.
 *
 *   3. Enhanced Cache Locality (Temporal Reuse):
 *      - The cache lines for row `i1` of matrices X, A, and B loaded during
 *        the forward sweep remain hot in L1/L2 cache for the boundary division
 *        and backward substitution, avoiding redundant DRAM fetches.
 * ============================================================================ */
static void kernel_adi(int tsteps,
                       int n,
                       DATA_TYPE POLYBENCH_2D(X, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(B, N, N, n, n))
{
  int t, i1, i2;
  
  /* Persistent thread team active across all simulation timesteps */
  #pragma omp parallel private(t, i1, i2)
  {
    for (t = 0; t < _PB_TSTEPS; t++)
    {
      /* ----------------------------------------------------------------------
       * Phase 1: Row-wise Tridiagonal System Solve (Fused Loop Kernel)
       * A single parallel loop schedules independent rows across threads.
       * Eliminates 2 intermediate barriers and maximizes L1/L2 cache residency.
       * ---------------------------------------------------------------------- */
      #pragma omp for
      for (i1 = 0; i1 < _PB_N; i1++)
      {
        /* Sub-step 1.1: Forward elimination along row i1 */
        for (i2 = 1; i2 < _PB_N; i2++)
        {
          X[i1][i2] = X[i1][i2] - X[i1][i2 - 1] * A[i1][i2] / B[i1][i2 - 1];
          B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2 - 1];
        }
        
        /* Sub-step 1.2: Boundary normalization on rightmost element */
        X[i1][_PB_N - 1] = X[i1][_PB_N - 1] / B[i1][_PB_N - 1];
      
        /* Sub-step 1.3: Backward substitution along row i1 */
        for (i2 = 0; i2 < _PB_N - 2; i2++)
        {
          X[i1][_PB_N - i2 - 2] = (X[i1][_PB_N - 2 - i2] - X[i1][_PB_N - 2 - i2 - 1] * A[i1][_PB_N - i2 - 3]) / B[i1][_PB_N - 3 - i2];
        }
      } 
          
      /* ----------------------------------------------------------------------
       * Phase 2: Column-wise Tridiagonal System Solve
       * Note: Outer loop represents the vertical recurrence, requiring inner
       * work-sharing across columns `i2`.
       * ---------------------------------------------------------------------- */
      /* Sub-step 2.1: Forward elimination along columns */
      for (i1 = 1; i1 < _PB_N; i1++)
      {
        #pragma omp for
        for (i2 = 0; i2 < _PB_N; i2++)
        {
          X[i1][i2] = X[i1][i2] - X[i1 - 1][i2] * A[i1][i2] / B[i1 - 1][i2];
          B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1 - 1][i2];
        }
      }
        
      /* Sub-step 2.2: Boundary normalization on bottom-most elements */
      #pragma omp for
      for (i2 = 0; i2 < _PB_N; i2++)
        X[_PB_N - 1][i2] = X[_PB_N - 1][i2] / B[_PB_N - 1][i2];
        
      /* Sub-step 2.3: Backward substitution along columns */
      for (i1 = 0; i1 < _PB_N - 2; i1++)
      {
        #pragma omp for
        for (i2 = 0; i2 < _PB_N; i2++)
          X[_PB_N - 2 - i1][i2] = (X[_PB_N - 2 - i1][i2] - X[_PB_N - i1 - 3][i2] * A[_PB_N - 3 - i1][i2]) / B[_PB_N - 2 - i1][i2];
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
