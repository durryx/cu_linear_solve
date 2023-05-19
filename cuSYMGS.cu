#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <ranges>
#include <algorithm>
#include <numeric>
#include <utility>
#include <set>
#include <assert.h>
#include <cmath>
#include <tuple>
// uncomment to disable assert()
// #define NDEBUG

/*
template <typename T, size_t solution_size>
auto get_different_results(const std::array<T, solution_size> &cpu_solution, const std::array<T, solution_size> &gpu_solution) -> std::vector<size_t>
{
    std::vector<size_t> err_indices;
    for (size_t i = 0; i < solution_size; i++)
    {
        if (!(cpu_solution[i] == gpu_solution[i]))
            err_indices.emplace_back(i);
    }

    return err_indices;
}

template <typename T, size_t solution_size>
auto get_error_distribution_data(const std::array<T, solution_size> &cpu_solution, const std::array<T, solution_size> &gpu_solution) -> std::tuple<double, double>
{
    std::multiset<double> elements_errors;
    for (auto &&[cpu_val, gpu_val] : std::views::zip(cpu_solution, gpu_solution))
    {
        double diff = abs(cpu_val - gpu_val);
        elements_errors.insert(diff);
    }

    double expected_value;
    std::vector<double> error_probability;
    for (auto &err : elements_errors)
    {
        double prob = elements_errors.count(err) / solution_size;
        error_probability.emplace_back(prob);
        expected_value += err * prob;
    }

    double variance;
    for (auto &&[prob, err] : std::views::zip(error_probability, elements_errors))
        variance += prob * pow(err - expected_value, 2);

    return {expected_value, variance};
}
*/

#define BLOCKN 512
#define THREADN 1024

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

double get_time()
{ // function to get the time of day in second
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(int **row_ptr, int **col_ind, float **values, float **matrixDiagonal, const char *filename, int *num_rows, int *num_cols, int *num_vals)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    // Get number of rows, columns, and non-zero values
    if (fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals) == EOF)
        printf("Error reading file");

    // printf("Rows: %d, Columns:%d, NNZ:%d\n", *num_rows, *num_cols, *num_vals);
    int *row_ptr_t = (int *)malloc((*num_rows + 1) * sizeof(int));
    int *col_ind_t = (int *)malloc(*num_vals * sizeof(int));
    float *values_t = (float *)malloc(*num_vals * sizeof(float));
    float *matrixDiagonal_t = (float *)malloc(*num_rows * sizeof(float));
    // Collect occurances of each row for determining the indices of row_ptr
    int *row_occurances = (int *)malloc(*num_rows * sizeof(int));
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

        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
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
void symgs_csr_sw(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *matrixDiagonal)
{

    // forward sweep
    for (int i = 0; i < num_rows; i++)
    {

        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        if (i == 143624)
            printf("SUM1_C: %f\n", sum);
        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];

            if (i == 143624)
            {
                printf("SUM_C: %f -- val %f -- x %f -- col_ind[j] %d \n", sum, values[j], x[col_ind[j]], col_ind[j]);
            }
        }
        if (i == 143624)
            printf("SUM2_C: %f\n", sum);

        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop
        if (i == 143624)
            printf("SUM3_C: %f\n", sum);

        x[i] = sum / currentDiagonal;
    }

    // backward sweep

    for (int i = num_rows - 1; i >= 0; i--)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        if (i == 1)
            printf("SUM B: %f\n", sum);

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];

            if (i == 1)
            {
                printf("SUM: %f -- val %f -- x %f -- col_ind[j] %d \n", sum, values[j], x[col_ind[j]], col_ind[j]);
            }
        }
        if (i == 1)
            printf("SUM4: %f\n", sum);

        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop
        if (i == 1)
            printf("SUM5: %f\n", sum);

        x[i] = sum / currentDiagonal;
    }
}

__global__ void cu_sweep_forward(const int *row_ptr, const int *col_ind, const float *mat, const int num_rows, float *vector, float *matrixDiagonal, int *dependant_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    dependant_rows[row] = -1;

    if (row < num_rows)
    {

        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        float sum = vector[row];
        float currentDiagonal = matrixDiagonal[row];

        for (int j = row_start; j < row_end; j++)
        {
            if (col_ind[j] < row)
            {
                dependant_rows[row] = row;
                return;
            }

            if (col_ind[j] < 0)
                continue;
        }

        for (int j = row_start; j < row_end; j++)
        {
            if (col_ind[j] < 0)
                continue;
            sum -= mat[j] * vector[col_ind[j]];
        }

        // Remove diagonal contribution from previous loop; see strictly minor indices in https://it.wikipedia.org/wiki/Metodo_di_Gauss-Seidel#Convergenza
        sum += vector[row] * currentDiagonal;

        // vector update
        vector[row] = sum / currentDiagonal;
    }
}

__global__ void cu_sweep_back(const int *row_ptr, const int *col_ind, const float *mat, const int num_rows, float *vector, float *matrixDiagonal, int *dependant_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    dependant_rows[row] = -1;

    if (row < num_rows)
    {

        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        float sum = vector[row];
        float currentDiagonal = matrixDiagonal[row];

        for (int j = row_start; j < row_end; j++)
        {
            if (col_ind[j] < 0)
                continue;
            if (col_ind[j] > row)
            {
                dependant_rows[row] = row;
                return;
            }
        }

        for (int j = row_start; j < row_end; j++)
        {
            if (col_ind[j] < 0)
                continue;
            sum -= mat[j] * vector[col_ind[j]];
        }

        sum += vector[row] * currentDiagonal;
        vector[row] = sum / currentDiagonal;
    }
}

int main(int argc, const char *argv[])
{
    /*
    if (argc != 2)
    {
        printf("Usage: ./exec matrix_file");
        return 0;
    }
    */
    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float *values;
    float *matrixDiagonal;

    // const char *filename = argv[1];
    const char *filename = "/home/gio/code/cuSYMGS/kmer_V4a.mtx";

    double start_cpu, end_cpu;
    double start_gpu, end_gpu;

    read_matrix(&row_ptr, &col_ind, &values, &matrixDiagonal, filename, &num_rows, &num_cols, &num_vals);
    float *x = (float *)malloc(num_rows * sizeof(float));
    float *xCopy = (float *)malloc(num_rows * sizeof(float));
    int *dependant_rows = (int *)malloc(num_rows * sizeof(int));

    // Generate a random vector
    srand(time(NULL));
    for (int i = 0; i < num_rows; i++)
    {
        x[i] = (float)(rand() % 100) / (float)(rand() % 100 + 1); // the number we use to divide cannot be 0, that's the reason of the +1
        xCopy[i] = x[i];
    }

    for (int i = 0; i < 10; i++)
    {
        printf("X : %f\n", x[i]);
    }

    // Compute in sw
    start_cpu = get_time();
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_cpu = get_time();

    printf("TEST \n %f \n", x[1]);

    // allocate space
    int *cu_row_ptr, *cu_col_ind;
    float *cu_mat, *cu_vector, *cu_matrixDiagonal;
    int *cu_dependant_rows;

    CHECK(cudaMalloc(&cu_row_ptr, (num_rows + 1) * sizeof(int)));
    CHECK(cudaMalloc(&cu_col_ind, num_vals * sizeof(int)));
    CHECK(cudaMalloc(&cu_mat, num_vals * sizeof(float)));
    CHECK(cudaMalloc(&cu_vector, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&cu_matrixDiagonal, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&cu_dependant_rows, num_rows * sizeof(int)));

    CHECK(cudaMemcpy(cu_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(cu_col_ind, col_ind, num_vals * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(cu_mat, values, num_vals * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(cu_vector, xCopy, num_rows * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(cu_matrixDiagonal, matrixDiagonal, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(BLOCKN, 1, 1);
    dim3 threadsPerBlock(THREADN, 1, 1);
    // compute in gpu
    start_gpu = get_time();

    int NUM_BLOCKS = (num_rows / 32) + 1;
    cu_sweep_forward<<<NUM_BLOCKS, 32>>>(
        cu_row_ptr,
        cu_col_ind,
        cu_mat,
        num_rows,
        cu_vector,
        cu_matrixDiagonal,
        cu_dependant_rows);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(xCopy, cu_vector, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(dependant_rows, cu_dependant_rows, num_rows * sizeof(int), cudaMemcpyDeviceToHost));

    // sort dependency array -- consider using a vector for dependant_rows
    constexpr int size = sizeof(dependant_rows) / sizeof(int);
    std::sort(dependant_rows, dependant_rows + size);
    auto non_negative = [](int &i)
    { return i >= 0; };
    auto first_dependant_row_itertor = std::find_if(dependant_rows, dependant_rows + size, non_negative);
    int first_dependant_row_index = first_dependant_row_itertor - dependant_rows;

    // dependat rows processing && to test cuda loop unroll performance or with thread pool queue
    for (int i = first_dependant_row_index; i < size; i++)
    {
        float sum = xCopy[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        if (i == 143624) // check if errors occure for other rows here
            printf("SUM1: %f\n", sum);

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * xCopy[col_ind[j]];

            if (i == 143624)
            {
                printf("SUM: %f -- val %f -- x %f -- col_ind[j] %d \n", sum, values[j], xCopy[col_ind[j]], col_ind[j]);
            }
        }

        sum += xCopy[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        if (i == 143624)
            printf("SUM3: %f\n", sum);

        xCopy[i] = sum / currentDiagonal;
    }

    // update cu_vector
    CHECK(cudaMemcpy(cu_vector, xCopy, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    // sweep back kernel
    cu_sweep_back<<<NUM_BLOCKS, 32>>>(
        cu_row_ptr,
        cu_col_ind,
        cu_mat,
        num_rows,
        cu_vector,
        cu_matrixDiagonal,
        cu_dependant_rows);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    // repeat same thing

    CHECK(cudaMemcpy(xCopy, cu_vector, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(dependant_rows, cu_dependant_rows, num_rows * sizeof(int), cudaMemcpyDeviceToHost));

    // sort
    std::sort(dependant_rows, dependant_rows + size);
    first_dependant_row_itertor = std::find_if(dependant_rows, dependant_rows + size, non_negative);
    first_dependant_row_index = first_dependant_row_itertor - dependant_rows;

    // dependant rows processing && to test cuda loop unroll performance or with thread pool queue
    for (int i = size - 1; i >= first_dependant_row_index; i--)
    {
        float sum = xCopy[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        if (i == 1) // check if errors occure for other rows here
            printf("SUM1 B2: %f\n", sum);

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * xCopy[col_ind[j]];

            if (i == 1)
            {
                printf("SUM: %f -- val %f -- x %f -- col_ind[j] %d \n", sum, values[j], xCopy[col_ind[j]], col_ind[j]);
            }
        }

        if (i == 1)
            printf("SUM4: %f\n", sum);

        sum += xCopy[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        if (i == 1)
            printf("SUM5: %f\n", sum);

        xCopy[i] = sum / currentDiagonal;
    }

    end_gpu = get_time();

    // error check
    int errors = 0;
    float maxError = 0.0;
    for (int i = 0; i < num_rows; i++)
    {
        if ((x[i] - xCopy[i] > 0.0001 || x[i] - xCopy[i] < -0.0001) && (x[i] - xCopy[i]) / x[i] > 0.001)
        {
            float err = x[i] - xCopy[i];
            err = err > 0 ? err : -err;
            maxError = err > maxError ? err : maxError;
            printf("\nerr %f -- %f, -- %d", x[i], xCopy[i], i);
            errors++;
        }
    }

    if (errors > 0)
        printf("Errors: %d\nMax error: %lf\n", errors, maxError);

    for (int i = 0; i < 100; i++)
    {
        printf("G %f \n", xCopy[i]);
        printf("C %f \n", x[i]);
    }

    // Print time
    printf("SYMGS Time CPU: %.10lf\n", end_cpu - start_cpu);
    printf("SYMGS Time GPU: %.10lf\n", end_gpu - start_gpu);

    // Free
    free(row_ptr);
    free(col_ind);
    free(values);
    free(matrixDiagonal);
    free(x);
    free(xCopy);

    CHECK(cudaFree(cu_row_ptr));
    CHECK(cudaFree(cu_col_ind));
    CHECK(cudaFree(cu_mat));
    CHECK(cudaFree(cu_vector));
    CHECK(cudaFree(cu_matrixDiagonal));
    CHECK(cudaFree(cu_dependant_rows));

    return 0;
}
