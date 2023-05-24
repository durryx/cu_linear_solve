#include "../src/gauss_seidel_sparse.cuh"
#include "../src/gauss_seidel_sparse.hpp"
#include <array>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

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

    csr_matrix matrix(filename);
    std::array<float, matrix.num_rows> vector;

    // Generate a random vector
    srand(time(NULL));
    for (int i = 0; i < matrix.num_rows; i++)
    {
        vector[i] =
            (float)(rand() % 100) /
            (float)(rand() % 100 + 1); // the number we use to divide cannot
                                       // be 0, that's the reason of the +1
    }

    int device;
    cudaGetDevice(&device);
    gauss_seidel_sparse_solve(matrix, vector, device)
}