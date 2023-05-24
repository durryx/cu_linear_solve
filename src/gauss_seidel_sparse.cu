#include "gauss_seidel_sparse.cuh"
#include "gauss_seidel_sparse.hpp"
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector>
// #include <helper_cuda.h>
#include <iostream>

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

__global__ void count_indipendant_rows(const int* row_ptr, const int* col_ind,
                                       const int num_rows,
                                       int* indipendant_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > num_rows)
        return;

    int row_start = row_ptr[row];
    if (col_ind[row_start] < 0)
    {
        return;
        // check whether to row_start++
    }
    if (col_ind[row_start] < row)
        atomicAdd(indipendant_rows, 1);
}

template <typename T>
__global__ void sweep_forward_all(const int* row_ptr, const int* col_ind,
                                  const T* matrix, const int num_rows,
                                  T* matrix_diagonal, T* vector,
                                  bool* dependant_locks)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > num_rows)
        return;

    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];
    T sum = vector[row];
    T current_diagonal = matrix_diagonal[row];

    for (int j = row_start; j < row_end; j++)
    {
        if (col_ind[j] < 0)
            continue;
        if (col_ind[j] < row && !dependant_locks[col_ind[j]])
            return;

        sum -= matrix[j] * vector[col_ind[j]];
    }

    sum += vector[row] * current_diagonal;
    vector[row] = sum / current_diagonal;
    dependant_locks[row] = true;
}

template <typename T>
__global__ void sweep_back_all(const int* row_ptr, const int* col_ind,
                               const T* matrix, const int num_rows,
                               T* matrix_diagonal, T* vector,
                               bool* dependant_locks)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > num_rows)
        return;

    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];
    T sum = vector[row];
    T current_diagonal = matrix_diagonal[row];

    for (int j = row_start; j < row_end; j++)
    {
        if (col_ind[j] < 0)
            continue;
        if (col_ind[j] > row && !dependant_locks[col_ind[j]])
            return;

        sum -= matrix[j] * vector[col_ind[j]];
    }

    sum += vector[row] * current_diagonal;
    vector[row] = sum / current_diagonal;
    dependant_locks[row] = true;
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

template <typename T>
void gauss_seidel_sparse_solve(csr_matrix matrix, std::vector<T> vector,
                               int device)
{
    int size = vector.size();
    int *dev_row_ptr, *dev_col_ind, *dev_ind_rows;
    T *dev_matrix, *dev_vector, *dev_matrix_diagonal;
    bool* dev_dependant_locks;

    CHECK(cudaMalloc(&dev_row_ptr, (matrix.num_rows + 1) * sizeof(int)));
    CHECK(cudaMalloc(&dev_col_ind, matrix.num_vals * sizeof(int)));
    CHECK(cudaMalloc(&dev_ind_rows, sizeof(int)));
    CHECK(cudaMalloc(&dev_matrix, matrix.num_vals * sizeof(T)));
    CHECK(cudaMalloc(&dev_vector, matrix.num_rows * sizeof(T)));
    CHECK(cudaMalloc(&dev_matrix_diagonal, matrix.num_rows * sizeof(T)));
    CHECK(cudaMalloc(&dev_dependant_locks, matrix.num_rows * sizeof(bool)));

    CHECK(cudaMemcpy(dev_row_ptr, matrix.row_ptr,
                     (matrix.num_rows + 1) * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_col_ind, matrix.col_ind, matrix.num_vals * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_matrix, matrix.values, matrix.num_vals * sizeof(T),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_vector, vector.data(), matrix.num_rows * sizeof(T),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_matrix_diagonal, matrix.matrix_diagonal,
                     matrix.num_rows * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dev_ind_rows, 0, sizeof(int)));
    CHECK(cudaMemset(dev_dependant_locks, 0, matrix.num_rows * sizeof(bool)));

    int driver_version = 0;
    int memory_pools = 0;
    cudaDeviceGetAttribute(&memory_pools, cudaDevAttrMemoryPoolsSupported,
                           device);
    cudaDriverGetVersion(&driver_version);

    int blocks = ceil(size / 128);
    dim3 threads_per_block(128, 1, 1);
    dim3 blocks_per_grid(blocks, 1, 1);

    if (driver_version < 11040 && !memory_pools || 1)
    {
        // cuda graph
    }
    else
    {
        count_indipendant_rows<<<blocks_per_grid, threads_per_block>>>(
            dev_row_ptr, dev_col_ind, size, dev_ind_rows);
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());

        int ind_rows = 0;
        CHECK(cudaMemcpy(&ind_rows, dev_ind_rows, sizeof(int),
                         cudaMemcpyDeviceToHost));
        // TO-FIX should check for null rows, they are indipendant
        int tot_iterations = matrix.num_rows - 2 * ind_rows;

        for (; tot_iterations >= 0; tot_iterations--)
        {
            sweep_forward_all<<<blocks_per_grid, threads_per_block>>>(
                dev_row_ptr, dev_col_ind, dev_matrix, size, dev_matrix_diagonal,
                vector.data(), dev_dependant_locks);
        }

        // kernel call to check if all locks == 1
    }

    CHECK(cudaMemcpy(vector.data(), dev_vector, matrix.num_rows * sizeof(T),
                     cudaMemcpyDeviceToHost));

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
