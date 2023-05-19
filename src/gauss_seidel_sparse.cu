#include "gauss_seidel_sparse.cuh"
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t err = call;                                          \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, \
                   __LINE__);                                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define CHECK_KERNELCALL()                                                     \
    {                                                                          \
        const cudaError_t err = cudaGetLastError();                            \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, \
                   __LINE__);                                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

struct csr_matrix
{
public:
    int* row_ptr;
    int* col_ind;
    float* values;
    float* matrix_diagonal;
    int num_rows;
    int num_cols;
    int num_vals;

    csr_matrix(const char* filename)
    {
        // copy read_matrix_function
    }

    virtual ~csr_matrix()
    {
        // destroy pointers
    }
};

// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row)
// format
void read_matrix(int** row_ptr, int** col_ind, float** values,
                 float** matrixDiagonal, const char* filename, int* num_rows,
                 int* num_cols, int* num_vals)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    // Get number of rows, columns, and non-zero values
    if (fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals) == EOF)
        printf("Error reading file");

    int* row_ptr_t = (int*)malloc((*num_rows + 1) * sizeof(int));
    int* col_ind_t = (int*)malloc(*num_vals * sizeof(int));
    float* values_t = (float*)malloc(*num_vals * sizeof(float));
    float* matrixDiagonal_t = (float*)malloc(*num_rows * sizeof(float));
    // Collect occurances of each row for determining the indices of row_ptr
    int* row_occurances = (int*)malloc(*num_rows * sizeof(int));
    for (int i = 0; i < *num_rows; i++)
    {
        row_occurances[i] = 0;
    }

    int row, column;
    float value;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF)
    {
        // Subtract 1 from row and column indices to match C format
        row--;
        column--;
        row_occurances[row]++;
    }

    // Set row_ptr
    int index = 0;
    for (int i = 0; i < *num_rows; i++)
    {
        row_ptr_t[i] = index;
        index += row_occurances[i];
    }
    row_ptr_t[*num_rows] = *num_vals;
    free(row_occurances);

    // Set the file position to the beginning of the file
    rewind(file);

    // Read the file again, save column indices and values
    for (int i = 0; i < *num_vals; i++)
    {
        col_ind_t[i] = -1;
    }

    if (fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals) == EOF)
        printf("Error reading file");

    int i = 0, j = 0;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF)
    {
        row--;
        column--;

        // Find the correct index (i + row_ptr_t[row]) using both row
        // information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1)
        {
            i++;
        }
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        if (row == column)
        {
            matrixDiagonal_t[j] = value;
            j++;
        }
        i = 0;
    }
    fclose(file);
    *row_ptr = row_ptr_t;
    *col_ind = col_ind_t;
    *values = values_t;
    *matrixDiagonal = matrixDiagonal_t;
}

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

__global__ void sweep_forward_all(const int* row_ptr, const int* col_ind,
                                  const float* matrix, const int num_rows,
                                  float* matrix_diagonal, float* vector,
                                  bool* dependant_locks)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > num_rows)
        return;

    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];
    float sum = vector[row];
    float current_diagonal = matrix_diagonal[row];

    while (true)
    {
        bool dependency_conflict = false;
        for (int j = row_start; j < row_end; j++)
        {
            if (col_ind[j] < 0)
                continue;
            if (col_ind[j] < row && !dependant_locks[col_ind[j]])
            {
                dependency_conflict = true;
                break;
            }
            sum -= matrix[j] * vector[col_ind[j]];
        }

        if (!dependency_conflict)
        {
            sum += vector[row] * current_diagonal;
            vector[row] = sum / current_diagonal;
            dependant_locks[row] = true;
            return;
        }

        // sync
    }
}

__global__ void sweep_back_all(const int* row_ptr, const int* col_ind,
                               const float* matrix_values, const int* num_rows,
                               float* matrix_diagonal)
{
}

__global__ void sweep_forward_decorporated(const int* row_ptr,
                                           const int* col_ind,
                                           const float* matrix_values,
                                           const int* num_rows,
                                           float* matrix_diagonal)
{
}

__global__ void sweep_back_decorporated(const int* row_ptr, const int* col_ind,
                                        const float* matrix_values,
                                        const int* num_rows,
                                        float* matrix_diagonal)
{
}

void gauss_seidel_sparse_solve()
{

    /*
    --- function for checking if all elements in cuda array is 0
    def


    --- incorporated in kernel if buffers supports all rows---
    array with to_process_rows = 0
    launch kernel:
        while(True)
            if is_processabale
                process and set index = 1
                return (or break)
            syncronize
        (process backward sweep same way?)


    --- decorporated from kernel if not enought memory ---
    use -1 or NaN for invalid
    todo_rows array = all rows
    while(True)
        (passing todo_rows) launch kernel:
            using (for all threads 1,2,3) get row index
            if(not valid)
                return
            if(row_can_be_completed)
                process and save set to 1 completed_indeces
                execute all_rows_compled_arrays
            return
        check if it is completed and end
        recalculate todo_rows array with completes_indices
        # if(not kernel: all_rows_are_completed_array)
        #     break


    --- ---
    function composition in cuda
    cuda graphs from decorporated method

    */
}
