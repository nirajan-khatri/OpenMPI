/*
 * matmul.c — MPI Parallel Matrix Multiplication
 * Strategy: Row-block distribution with MPI_Gatherv
 *
 * Author: [Nirajan Khatri]
 *
 * External sources used:
 * - row_wise_matrix_mult.c provided on Moodle (fillArray, my_rand, concatenate)
 * - AI tool (Gemini) used for code structure guidance
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

double my_rand(unsigned long *state, double lower, double upper)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    unsigned long x = (*state * 0x2545F4914F6CDD1DULL);
    const double inv = 1.0 / (double)(1ULL << 53);
    double u = (double)(x >> 11) * inv;
    return lower + (upper - lower) * u;
}

unsigned concatenate(unsigned x, unsigned y)
{
    unsigned pow = 10;
    while (y >= pow)
        pow *= 10;
    return x * pow + y;
}

static void fill_rows(double *arr, int n, int row_start, int num_rows,
                       int seed, int value)
{
    for (int i = 0; i < num_rows; i++) {
        int global_row = row_start + i;
        for (int j = 0; j < n; j++) {
            unsigned long state = concatenate(global_row, j) + seed + value;
            arr[i * n + j] = my_rand(&state, 0, 1);
        }
    }
}

static void fill_matrix(double *arr, int n, int seed, int value)
{
    fill_rows(arr, n, 0, n, seed, value);
}

static void print_matrix(const char *label, const double *arr, int n)
{
    printf("%s:\n", label);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f", arr[i * n + j]);
            if (j < n - 1) printf(" ");
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0)
            fprintf(stderr, "Usage: mpirun -np <p> ./matmul n seed verbose\n");
        MPI_Finalize();
        return 1;
    }

    int n       = atoi(argv[1]);
    int seed    = atoi(argv[2]);
    int verbose = atoi(argv[3]);

    double t_start = MPI_Wtime();

    int base_rows    = n / size;
    int remainder    = n % size;
    int my_num_rows  = base_rows + (rank < remainder ? 1 : 0);
    int my_start_row = rank * base_rows + (rank < remainder ? rank : remainder);

    double *local_A = (double *)malloc((size_t)my_num_rows * n * sizeof(double));
    double *B       = (double *)malloc((size_t)n * n * sizeof(double));
    double *local_C = (double *)calloc((size_t)my_num_rows * n, sizeof(double));

    if (!local_A || !B || !local_C) {
        fprintf(stderr, "Rank %d: memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fill_rows(local_A, n, my_start_row, my_num_rows, seed, 0);
    fill_matrix(B, n, seed, 1);

    for (int i = 0; i < my_num_rows; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = local_A[i * n + k];
            for (int j = 0; j < n; j++) {
                local_C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }

    free(local_A);
    free(B);

    int *recv_counts = NULL;
    int *displs      = NULL;
    double *C        = NULL;

    if (rank == 0) {
        recv_counts = (int *)malloc(size * sizeof(int));
        displs      = (int *)malloc(size * sizeof(int));
        C           = (double *)malloc((size_t)n * n * sizeof(double));
        if (!recv_counts || !displs || !C) {
            fprintf(stderr, "Rank 0: memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int offset = 0;
        for (int r = 0; r < size; r++) {
            int rows_r = base_rows + (r < remainder ? 1 : 0);
            recv_counts[r] = rows_r * n;
            displs[r]      = offset;
            offset        += recv_counts[r];
        }
    }

    MPI_Gatherv(local_C, my_num_rows * n, MPI_DOUBLE,
                C, recv_counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (verbose == 1 && n <= 10) {
            double *A_full = (double *)malloc((size_t)n * n * sizeof(double));
            double *B_full = (double *)malloc((size_t)n * n * sizeof(double));
            fill_matrix(A_full, n, seed, 0);
            fill_matrix(B_full, n, seed, 1);

            print_matrix("Matrix A", A_full, n);
            printf("\n");
            print_matrix("Matrix B", B_full, n);
            printf("\n");
            print_matrix("Matrix C (Result)", C, n);
            printf("\n");

            free(A_full);
            free(B_full);
        }

        double checksum = 0.0;
        for (int i = 0; i < n * n; i++) {
            checksum += C[i];
        }
        printf("Checksum: %f\n", checksum);

        double t_end = MPI_Wtime();
        printf("Execution time with %d ranks: %.2f s\n", size, t_end - t_start);

        free(C);
        free(recv_counts);
        free(displs);
    }

    free(local_C);

    MPI_Finalize();
    return 0;
}
