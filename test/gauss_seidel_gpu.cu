#include "../src/gauss_seidel_sparse.cpp"
#include "../src/gauss_seidel_sparse.cu"
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <vector>

// compile with -DDEBUG_MODE=1 or define this macro variable
// #define DEBUG_MODE 1

typedef std::chrono::high_resolution_clock::time_point time_var;

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
    if (DEBUG_OUTPUT)
    {
        std::cout << matrix.num_rows << '\n';
    }

    // Generate a random vector
    std::default_random_engine eng{};
    std::uniform_real_distribution<> dist{1, 100};
    auto gen_random = [&]() { return dist(eng); };

    // std::for_each(vector.begin(), vector.end(), gen_random());
    srand(time(NULL));
    for (int i = 0; i < matrix.num_rows; i++)
        vector[i] = (float)gen_random();
    std::vector<float> cpu_sol = vector;
    std::vector<float> gpu_sol = vector;

    time_var t1 = std::chrono::high_resolution_clock::now();
    symgs_csr_sw(matrix, cpu_sol);
    double cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::high_resolution_clock::now() - t1)
                          .count();

    int device;
    cudaGetDevice(&device);
    gauss_seidel_sparse_solve(matrix, gpu_sol, device);
    double gpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::high_resolution_clock::now() - t1)
                          .count();

    auto err_indices = get_different_results(cpu_sol, gpu_sol);

    std::cout << "errors: " << err_indices.size() << "\n";
    std::cout << "cpu_time: " << cpu_time << "ns"
              << "\n";
    std::cout << "gpu_time: " << gpu_time << "ns"
              << "\n";
    std::cout << "cpu_time/gpu_time ratio: " << cpu_time / gpu_time
              << std::endl;
}