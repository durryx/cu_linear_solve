#include <algorithm>
#include <array>
#include <iostream>

template <size_t n>
__global__ void lock_check(const int num_rows, const bool* dependant_locks,
                           int* not_terminated)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

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
            // *not_terminated = true;
        }

        if (threadIdx.x == 0 && *not_terminated)
            warp_found = true;

        i++;
        __syncthreads();
    }
}

int main(int argc, const char* argv[])
{
    bool check[51813503] = {0};

    check[27138944] = false;
    check[27139152] = false;
    check[27139200] = false;
    check[27139206] = false;
    check[27139207] = false;
    check[27139208] = false;
    check[27139209] = false;
    check[27139210] = false;
    check[27139211] = false;
    check[27139212] = false;

    bool* dev_check;
    int* dev_not_terminated;
    cudaMalloc(&dev_check, 51813503 * sizeof(bool));
    cudaMemcpy(dev_check, &check[0], 51813503 * sizeof(bool),
               cudaMemcpyHostToDevice);

    int not_terminated = false;
    cudaMalloc(&dev_not_terminated, sizeof(int));
    cudaMemcpy(dev_not_terminated, &not_terminated, sizeof(int),
               cudaMemcpyHostToDevice);

    dim3 threads_per_block(128, 1, 1);
    constexpr size_t n = 2;
    int blocks = ceil(51813503 / (n * 128)) + 1;
    dim3 blocks_per_grid(blocks, 1, 1);

    lock_check<n><<<blocks_per_grid, threads_per_block>>>(51813503, dev_check,
                                                          dev_not_terminated);

    cudaMemcpy(&not_terminated, dev_not_terminated, sizeof(int),
               cudaMemcpyDeviceToHost);
    if (not_terminated)
        printf("detected\n");
    else
        printf("not detected\n");

    return 0;
}
