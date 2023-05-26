#include "../src/gauss_seidel_sparse.cpp"
#include "../src/gauss_seidel_sparse.cu"
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <vector>

int main(int argc, const char* argv[])
{
    if (argc != 2)
    {
        printf("Usage: ./exec matrix_file");
        return 0;
    }
    const char* filename = argv[1];

    csr_matrix matrix(filename);
    std::vector<float> vector(matrix.num_rows, .0F);

    // Generate a random vector
    std::default_random_engine eng{};
    std::uniform_real_distribution<> dist{1, 100};
    auto gen_random = [&]() { return dist(eng); };

    srand(time(NULL));
    for (int i = 0; i < matrix.num_rows; i++)
    {
        vector[i] = (float)gen_random();
    }

    std::vector<float> cpu_sol;
    cpu_sol = vector;
    symgs_csr_sw(matrix, cpu_sol);

    std::vector<float> gpu_sol;
    gpu_sol = vector;
    int device;
    cudaGetDevice(&device);
    gauss_seidel_sparse_solve(matrix, gpu_sol, device);

    auto err_indices = get_different_results(cpu_sol, gpu_sol);
    std::cout << err_indices.size() << '\n';
}