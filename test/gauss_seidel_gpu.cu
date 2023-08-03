#include "../src/gauss_seidel_sparse.cpp"
#include "../src/gauss_seidel_sparse.cu"
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <type_traits>
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

    // Generate a random vector
    std::default_random_engine eng{};
    std::uniform_real_distribution<> dist{1, 100};
    auto gen_random = [&dist, &eng]() mutable { return (float)dist(eng); };

    srand(time(NULL));
    // std::transform(vector.begin(), vector.end(), vector.begin(), gen_random);
    std::generate(vector.begin(), vector.end(), gen_random);
    std::vector<float> cpu_sol = vector;
    std::vector<float> gpu_sol = vector;

    time_var t1 = std::chrono::high_resolution_clock::now();
    symgs_csr_sw(matrix, cpu_sol);
    double cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::high_resolution_clock::now() - t1)
                          .count();

    int device;
    cudaGetDevice(&device);
    gauss_seidel_sparse_solve<float, 10>(matrix, gpu_sol, device);
    double gpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::high_resolution_clock::now() - t1)
                          .count();

    auto err_indices = get_different_results(cpu_sol, gpu_sol);
    if (DEBUG_MODE && err_indices.size() != 0)
        dump_errors(cpu_sol, gpu_sol, err_indices, 100, "detailed errors");

    std::cout << "errors: " << err_indices.size() << "\n"
              << "cpu_time: " << cpu_time << "ns"
              << "\n"
              << "gpu_time: " << gpu_time << "ns"
              << "\n"
              << "cpu_time/gpu_time ratio: " << cpu_time / gpu_time
              << std::endl;
}