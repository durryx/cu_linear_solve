#include "gauss_seidel_sparse.cuh"
#include <array>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
// #include <helper_cuda.h>
#include <iostream>
#include <set>
#include <tuple>
#include <unordered_set>
#include <vector>

// helper functions and utilities to work with CUDA

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

csr_matrix::csr_matrix(const char* filename) {}

csr_matrix::~csr_matrix() {}

auto get_max_iterations(csr_matrix matrix)
{
    std::set<size_t> sf_dependant_idx;
    std::set<size_t> sb_dependant_idx;

    for (int i = 0; i > matrix.num_rows; i++)
    {
        const int row_end = matrix.row_ptr[i + 1];
        const int row_start = matrix.row_ptr[i];

        for (int j = row_start; j < row_end; j++)
        {
            if (matrix.col_ind[j] < i)
                sf_dependant_idx.insert(matrix.col_ind[j]);
            else
                break;
            // check if also sweeb back counter is needed
        }
    }
    return std::tuple{sf_dependant_idx.size(),
                      matrix.num_rows - sf_dependant_idx.size()};
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

__global__ void sweep_back_all(const int* row_ptr, const int* col_ind,
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

template <typename T, size_t size>
void gauss_seidel_sparse_solve(csr_matrix matrix, std::array<T, size> vector,
                               int device)
{

    int *dev_row_ptr, *dev_col_ind;
    T *dev_matrix, *dev_vector, *dev_matrix_diagonal;
    bool* dev_dependant_locks;

    int driver_version = 0;
    int memory_pools = 0;
    cudaDeviceGetAttribute(&memory_pools, cudaDevAttrMemoryPoolsSupported,
                           device);
    cudaDriverGetVersion(&driver_version);

    constexpr int blocks = ceil(size / 128);
    dim3 threads_per_block(128, 1, 1);
    dim3 blocks_per_grid(blocks, 1, 1);

    if (driver_version < 11040 && !memory_pools)
    {
        // cuda graph
    }
    else
    {
        sweep_forward_all<<<blocks_per_grid, threads_per_block>>>(
            dev_row_ptr, dev_col_ind, dev_matrix, size, dev_matrix_diagonal,
            vector.data(), dev_dependant_locks);

        // kernel call to check if all locks == 1
    }

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
