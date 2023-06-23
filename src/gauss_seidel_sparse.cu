#include "gauss_seidel_sparse.cuh"
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

/*
set terminated to true
after this function check if terminated is true
else keep iterating
__global__ void check_row_locks(const int* row_ptr, const int* col_ind,
                                const int num_rows, bool* dependant_locks,
                                bool* terminated)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > num_rows)
        return;

    if (dependant_locks[row] == false)
    {
        atomicExch(terminated, 0);
        __threadfence();
        asm("trap;");
    }
}
*/

template <typename T, size_t n>
__global__ void rows_lock_check(const int num_rows, const bool* dependant_locks,
                                volatile bool* not_terminated)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    volatile __shared__ bool warp_found;
    if (threadIdx.x == 0)
        warp_found = *not_terminated;
    __syncthreads();

    size_t i = 0;
    while (!warp_found && i < n)
    {
        if (index + i >= num_rows)
            return;

        if (dependant_locks[index + i] == false)
        {
            warp_found = true;
            *not_terminated = true;
        }

        if (threadIdx.x == 0 && *not_terminated)
            warp_found = true;

        i++;
        __syncthreads();
    }
}

// process more than one
template <typename T, size_t n>
__global__ void sweep_forward_n(const int* row_ptr, const int* col_ind,
                                const T* matrix, const int num_rows,
                                const T* matrix_diagonal, T* vector,
                                bool* dependant_locks)
{
    int index = n * (blockIdx.x * blockDim.x + threadIdx.x);

    for (size_t i = 0; i < n; i++)
    {
        if (index + i >= num_rows)
            return;

        if (dependant_locks[index + i] == true)
            continue;

        int row_start = row_ptr[index + i];
        int row_end = row_ptr[index + i + 1];
        T sum = vector[index + i];
        T current_diagonal = matrix_diagonal[index + i];

        bool skip_row = false;
        for (int j = row_start; j < row_end; j++)
        {
            if (col_ind[j] < 0)
                continue;
            if (col_ind[j] < (index + i) && !dependant_locks[col_ind[j]])
            {
                skip_row = true;
                break;
            }
            sum -= matrix[j] * vector[col_ind[j]];
        }
        if (skip_row)
            continue;

        sum += vector[index + i] * current_diagonal;
        vector[index + i] = sum / current_diagonal;
        dependant_locks[index + i] = true;
    }
}

template <typename T, size_t n>
__global__ void sweep_back_n(const int* row_ptr, const int* col_ind,
                             const T* matrix, const int num_rows,
                             const T* matrix_diagonal, T* vector,
                             bool* dependant_locks)
{
    int index = n * (blockIdx.x * blockDim.x + threadIdx.x);

    for (size_t i = 0; i < n; i++)
    {
        if (index + i >= num_rows)
            return;

        if (dependant_locks[index + i] == true)
            continue;

        int row_start = row_ptr[index + i];
        int row_end = row_ptr[index + i + 1];
        T sum = vector[index + i];
        T current_diagonal = matrix_diagonal[index + i];

        bool skip_row = false;
        for (int j = row_end - 1; j < row_start; j--)
        {
            if (col_ind[j] < 0)
                continue;
            if (col_ind[j] > (index + i) && !dependant_locks[col_ind[j]])
            {
                skip_row = true;
                break;
            }
            sum -= matrix[j] * vector[col_ind[j]];
        }
        if (skip_row)
            continue;

        sum += vector[index + i] * current_diagonal;
        vector[index + i] = sum / current_diagonal;
        dependant_locks[index + i] = true;
    }
}

template <typename T>
void gauss_seidel_sparse_solve(csr_matrix& matrix, std::vector<T>& vector,
                               int device)
{
    int *dev_row_ptr, *dev_col_ind;
    T *dev_matrix, *dev_vector, *dev_matrix_diagonal;
    bool *dev_dependant_locks, *dev_not_terminated;

    CHECK(cudaMalloc(&dev_row_ptr, (matrix.num_rows + 1) * sizeof(int)));
    CHECK(cudaMalloc(&dev_col_ind, matrix.num_vals * sizeof(int)));
    CHECK(cudaMalloc(&dev_matrix, matrix.num_vals * sizeof(T)));
    CHECK(cudaMalloc(&dev_vector, matrix.num_rows * sizeof(T)));
    CHECK(cudaMalloc(&dev_matrix_diagonal, matrix.num_rows * sizeof(T)));
    CHECK(cudaMalloc(&dev_dependant_locks, matrix.num_rows * sizeof(bool)));
    CHECK(cudaMalloc(&dev_not_terminated, sizeof(bool)));

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
    CHECK(cudaMemset(dev_dependant_locks, 0, matrix.num_rows * sizeof(bool)));

    int driver_version = 0;
    int memory_pools = 0;
    cudaDeviceGetAttribute(&memory_pools, cudaDevAttrMemoryPoolsSupported,
                           device);
    cudaDriverGetVersion(&driver_version);

    constexpr size_t n = 1;
    int blocks = ceil(matrix.num_rows / (n * 128));
    dim3 threads_per_block(128, 1, 1);
    dim3 blocks_per_grid(blocks, 1, 1);

    if (driver_version < 11040 && !memory_pools && 0)
    {
        // cuda graph
    }
    else
    {
        bool not_terminated = true;
        while (not_terminated)
        {
            not_terminated = false;
            CHECK(cudaMemset(dev_not_terminated, false, sizeof(bool)));

            sweep_forward_n<T, n><<<blocks_per_grid, threads_per_block>>>(
                dev_row_ptr, dev_col_ind, dev_matrix, matrix.num_rows,
                dev_matrix_diagonal, dev_vector, dev_dependant_locks);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());

            rows_lock_check<T, n><<<blocks_per_grid, threads_per_block>>>(
                matrix.num_rows, dev_dependant_locks, dev_not_terminated);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(&not_terminated, dev_not_terminated, sizeof(bool),
                             cudaMemcpyDeviceToHost));
        }

        if (DEBUG_MODE)
        {
            // dump not working
            CHECK(cudaMemcpy(&vector[0], dev_vector,
                             matrix.num_rows * sizeof(T),
                             cudaMemcpyDeviceToHost));
            dump_vector(vector, 100, "nvidia mode");
        }

        CHECK(
            cudaMemset(dev_dependant_locks, 0, matrix.num_rows * sizeof(bool)));
        not_terminated = true;

        while (not_terminated)
        {
            not_terminated = false;
            CHECK(cudaMemset(dev_not_terminated, false, sizeof(bool)));

            sweep_back_n<T, n><<<blocks_per_grid, threads_per_block>>>(
                dev_row_ptr, dev_col_ind, dev_matrix, matrix.num_rows,
                dev_matrix_diagonal, dev_vector, dev_dependant_locks);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());

            rows_lock_check<T, n><<<blocks_per_grid, threads_per_block>>>(
                matrix.num_rows, dev_dependant_locks, dev_not_terminated);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(&not_terminated, dev_not_terminated, sizeof(bool),
                             cudaMemcpyDeviceToHost));
        }
    }

    CHECK(cudaMemcpy(vector.data(), dev_vector, matrix.num_rows * sizeof(T),
                     cudaMemcpyDeviceToHost));
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
