#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include "../src/gauss_seidel_sparse.cuh"


double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}


int main(int argc, const char* argv[])
{
    if (argc != 2)
    {
        printf("Usage: ./exec matrix_file");
        return 0;
    }

    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float* values;
    float* matrixDiagonal;

    const char* filename = argv[1];

    double start_cpu, end_cpu;
    double start_gpu, end_gpu;

    read_matrix(&row_ptr, &col_ind, &values, &matrixDiagonal, filename,
                &num_rows, &num_cols, &num_vals);
    float* x = (float*)malloc(num_rows * sizeof(float));
    float* xCopy = (float*)malloc(num_rows * sizeof(float));

    // Generate a random vector
    srand(time(NULL));
    for (int i = 0; i < num_rows; i++)
    {
        x[i] = (float)(rand() % 100) /
               (float)(rand() % 100 + 1); // the number we use to divide cannot
                                          // be 0, that's the reason of the +1
        xCopy[i] = x[i];
    }

    // ############################ cpu part ############################
    start_cpu = get_time();
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_cpu = get_time();

    // ############################ cpu part ############################
}