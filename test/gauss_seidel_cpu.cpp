#include "../src/gauss_seidel_sparse.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <vector>

// CPU implementation of SYMGS using CSR, DO NOT CHANGE THIS
void symgs_csr_sw(const int* row_ptr, const int* col_ind, const float* values,
                  const int num_rows, float* x, float* matrixDiagonal)
{

    // forward sweep
    for (int i = 0; i < num_rows; i++)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }

        sum +=
            x[i] *
            currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }

    // backward sweep
    for (int i = num_rows - 1; i >= 0; i--)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }
        sum +=
            x[i] *
            currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }
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

    csr_matrix matrix(filename);
    std::vector<float> vector(matrix.num_rows);

    // Generate a random vector
    srand(time(NULL));
    for (int i = 0; i < matrix.num_rows; i++)
    {
        vector[i] =
            (float)(rand() % 100) /
            (float)(rand() % 100 + 1); // the number we use to divide cannot
                                       // be 0, that's the reason of the +1
    }

    symgs_csr_sw(matrix.row_ptr, matrix.col_ind, matrix.values, matrix.num_rows,
                 vector.data(), matrix.matrix_diagonal);
}