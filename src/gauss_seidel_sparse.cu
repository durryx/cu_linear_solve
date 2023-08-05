#include "gauss_seidel_sparse.cuh"
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector>
// #include <helper_cuda.h>
#include <algorithm>
#include <iostream>
#include <optional>
#include <type_traits>

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

// use this to calculate active warps using registers sharing
// recast bool/int array to a bigger value and check if == 0 see if can create
// pointer of certain bytes here
// memory coalescence

// template <typename T, size_t size>
// __constant__ T vector_copy_test[size];

template <typename T, size_t n>
__global__ void rows_lock_check(const int num_rows, const bool* dependant_locks,
                                int* not_terminated)
{
    int index = n * (blockIdx.x * blockDim.x + threadIdx.x);

    volatile __shared__ int warp_found;
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
            // atomicExch(&warp_found, 1);
            warp_found = true;
            atomicCAS(not_terminated, 0, true);
            // assert(retval != true);
            //*not_terminated = true;
        }

        if (threadIdx.x == 0 && *not_terminated)
            warp_found = true;

        i++;
        if constexpr (n > 1)
            __syncthreads();
    }
}

// check if warp is empty or calculate minimum row dependency index with
// recduction
template <typename T, size_t n>
__global__ void sweep_forward_n(const int* row_ptr, const int* col_ind,
                                const T* matrix, const int num_rows,
                                const T* matrix_diagonal, T* vector,
                                T* vector_copy, bool* dependant_locks)
{
    int index = n * (blockIdx.x * blockDim.x + threadIdx.x);

    /*
    volatile __shared__ int warp_idle;
    if (threadIdx.x == 0)
        warp_idle = 0;
    __syncthreads();
    */

    for (size_t i = 0; i < n; i++)
    {
        if (index + i >= num_rows)
            return;

        if (dependant_locks[index + i] == true)
            continue;

        int row_start = row_ptr[index + i];
        int row_end = row_ptr[index + i + 1];
        T sum = vector_copy[index + i];

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
            if (col_ind[j] < index + i)
                sum -= matrix[j] * vector[col_ind[j]];
            else
                sum -= matrix[j] * vector_copy[col_ind[j]];
        }

        if (skip_row)
        {
            continue;
            /*
            write to warp_idle
            */
        }

        T current_diagonal = matrix_diagonal[index + i];
        sum += vector_copy[index + i] * current_diagonal;
        vector[index + i] = sum / current_diagonal;
        dependant_locks[index + i] = true;
    }
}

template <typename T, size_t n>
__global__ void sweep_back_n(const int* row_ptr, const int* col_ind,
                             const T* matrix, const int num_rows,
                             const T* matrix_diagonal, T* vector,
                             T* vector_copy, bool* dependant_locks)
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
        T sum = vector_copy[index + i];

        bool skip_row = false;
        for (int j = row_end - 1; j >= row_start; j--)
        {
            if (col_ind[j] < 0)
                continue;
            if (col_ind[j] > (index + i))
            {
                if (!dependant_locks[col_ind[j]])
                {
                    skip_row = true;
                    break;
                }
                else
                    sum -= matrix[j] * vector[col_ind[j]];
            }
            sum -= matrix[j] * vector_copy[col_ind[j]];
        }
        if (skip_row)
            continue;

        T current_diagonal = matrix_diagonal[index + i];
        sum += vector_copy[index + i] * current_diagonal;
        vector[index + i] = sum / current_diagonal;
        dependant_locks[index + i] = true;
    }
}

template <typename T, size_t conseq_iterations>
void gauss_seidel_sparse_solve(csr_matrix& matrix, std::vector<T>& vector,
                               int device)
{
    int *dev_row_ptr, *dev_col_ind;
    int* dev_not_terminated;
    T *dev_matrix, *dev_vector, *dev_matrix_diagonal, *dev_vector_copy;
    bool* dev_dependant_locks;

    CHECK(cudaMalloc(&dev_row_ptr, (matrix.num_rows + 1) * sizeof(int)));
    CHECK(cudaMalloc(&dev_col_ind, matrix.num_vals * sizeof(int)));
    CHECK(cudaMalloc(&dev_matrix, matrix.num_vals * sizeof(T)));
    CHECK(cudaMalloc(&dev_vector, matrix.num_rows * sizeof(T)));
    CHECK(cudaMalloc(&dev_vector_copy, matrix.num_rows * sizeof(T)));
    CHECK(cudaMalloc(&dev_matrix_diagonal, matrix.num_rows * sizeof(T)));
    CHECK(cudaMalloc(&dev_dependant_locks, matrix.num_rows * sizeof(bool)));
    CHECK(cudaMalloc(&dev_not_terminated, sizeof(int)));

    CHECK(cudaMemcpy(dev_row_ptr, matrix.row_ptr,
                     (matrix.num_rows + 1) * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_col_ind, matrix.col_ind, matrix.num_vals * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_matrix, matrix.values, matrix.num_vals * sizeof(T),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_vector, vector.data(), matrix.num_rows * sizeof(T),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_vector_copy, vector.data(),
                     matrix.num_rows * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_matrix_diagonal, matrix.matrix_diagonal,
                     matrix.num_rows * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dev_dependant_locks, 0, matrix.num_rows * sizeof(bool)));

    // CHECK(cudaMemcpyToSymbol(vector_copy_test<T, 5>, vector.data(),
    //                          matrix.num_rows * sizeof(T)));

    int driver_version = 0;
    int memory_pools = 0;
    cudaDeviceGetAttribute(&memory_pools, cudaDevAttrMemoryPoolsSupported,
                           device);
    cudaDriverGetVersion(&driver_version);

    constexpr size_t n = 4;
    int blocks = ceil((float)matrix.num_rows / (n * 128));
    // int blocks = (matrix.num_rows + (n*128) - 1)/(n*128);
    dim3 threads_per_block(128, 1, 1);
    dim3 blocks_per_grid(blocks, 1, 1);

    if (driver_version < 11040 && !memory_pools && 0)
    {
        // cuda graph
    }
    else
    {
        for (size_t i = 0; i <= conseq_iterations; i++)
        {
            sweep_forward_n<T, n><<<blocks_per_grid, threads_per_block>>>(
                dev_row_ptr, dev_col_ind, dev_matrix, matrix.num_rows,
                dev_matrix_diagonal, dev_vector, dev_vector_copy,
                dev_dependant_locks);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());
        }

        int not_terminated = false;
        rows_lock_check<T, n><<<blocks_per_grid, threads_per_block>>>(
            matrix.num_rows, dev_dependant_locks, dev_not_terminated);
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&not_terminated, dev_not_terminated, sizeof(int),
                         cudaMemcpyDeviceToHost));

        while (not_terminated)
        {
            not_terminated = false;
            CHECK(cudaMemset(dev_not_terminated, false, sizeof(int)));

            sweep_forward_n<T, n><<<blocks_per_grid, threads_per_block>>>(
                dev_row_ptr, dev_col_ind, dev_matrix, matrix.num_rows,
                dev_matrix_diagonal, dev_vector, dev_vector_copy,
                dev_dependant_locks);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());

            rows_lock_check<T, n><<<blocks_per_grid, threads_per_block>>>(
                matrix.num_rows, dev_dependant_locks, dev_not_terminated);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(&not_terminated, dev_not_terminated, sizeof(int),
                             cudaMemcpyDeviceToHost));
        }

        static_assert(
            std::is_same_v<decltype(vector[0]), decltype(*dev_vector)> == true);

        CHECK(cudaMemcpy(dev_vector_copy, dev_vector, sizeof(T) * vector.size(),
                         cudaMemcpyDeviceToDevice))
        CHECK(
            cudaMemset(dev_dependant_locks, 0, matrix.num_rows * sizeof(bool)));

        for (size_t i = 0; i <= conseq_iterations; i++)
        {

            sweep_back_n<T, n><<<blocks_per_grid, threads_per_block>>>(
                dev_row_ptr, dev_col_ind, dev_matrix, matrix.num_rows,
                dev_matrix_diagonal, dev_vector, dev_vector_copy,
                dev_dependant_locks);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());
        }
        not_terminated = false;
        rows_lock_check<T, n><<<blocks_per_grid, threads_per_block>>>(
            matrix.num_rows, dev_dependant_locks, dev_not_terminated);
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&not_terminated, dev_not_terminated, sizeof(int),
                         cudaMemcpyDeviceToHost));

        while (not_terminated)
        {
            not_terminated = false;
            CHECK(cudaMemset(dev_not_terminated, false, sizeof(int)));

            sweep_back_n<T, n><<<blocks_per_grid, threads_per_block>>>(
                dev_row_ptr, dev_col_ind, dev_matrix, matrix.num_rows,
                dev_matrix_diagonal, dev_vector, dev_vector_copy,
                dev_dependant_locks);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());

            rows_lock_check<T, n><<<blocks_per_grid, threads_per_block>>>(
                matrix.num_rows, dev_dependant_locks, dev_not_terminated);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(&not_terminated, dev_not_terminated, sizeof(int),
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
    # if(not kernel: all_rows_are_completed_array)
    #     break

*/
