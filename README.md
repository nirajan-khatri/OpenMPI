# MPI Parallel Matrix Multiplication — Exam 2

## Compilation

```bash
# Load required modules (Fulda HPC Cluster)
module load gcc/14.3.0
module load openmpi

# Compile
make clean && make
```

## Program: matmul (Row-Block Distribution)

Each MPI rank generates its own rows of A and the full matrix B locally (no broadcast needed).
Each rank computes its block of C = A × B using an i-k-j loop for cache efficiency.
Rank 0 gathers all blocks via `MPI_Gatherv`.

Works with any number of processes.

```bash
mpirun -np <num_processes> ./matmul n seed verbose
```

### Parameters

| Parameter | Description                                                       |
| --------- | ----------------------------------------------------------------- |
| `n`       | Size of the square matrices (n × n)                               |
| `seed`    | Initial value for the random number generator                     |
| `verbose` | `1` = print matrices A, B, C if n ≤ 10; `0` = print only checksum |

## Verification

```bash
# Expected checksum: 17.502887
mpirun -np 4 ./matmul 4 42 1
mpirun -np 4 ./matmul 4 42 0
mpirun -np 1 ./matmul 4 42 1
```

## Speedup Measurements

Submit Slurm script for n = 8000, seed = 42, verbose = 0:

```bash
sbatch run_matmul.sh
```

Runs on 1, 2, 4, 6, 8 nodes with 64 MPI processes per node.
Each configuration is run 3 times for averaging.

## File Structure

| File                       | Description                                      |
| -------------------------- | ------------------------------------------------ |
| `matmul.c`                 | Row-block distribution MPI matrix multiplication |
| `row_wise_matrix_mult.c`   | Reference implementation (from Moodle)           |
| `Makefile`                 | Build system                                     |
| `run_matmul.sh`            | Slurm script for speedup measurements            |
| `performance_analysis.tex` | LaTeX source for performance analysis PDF        |
| `README.md`                | This file                                        |

## External Sources

- `row_wise_matrix_mult.c` provided on Moodle (`my_rand`, `concatenate` functions)
- AI tool (Gemini) used for code structure guidance
