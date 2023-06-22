#include "gauss_seidel_sparse.hpp"
#include <cassert>
#include <iostream>
#include <set>
#include <tuple>
#include <vector>

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <numeric>
#include <ranges>
#include <set>
#include <tuple>
#include <utility>

// uncomment to disable assert()
// #define NDEBUG

template <typename T>
auto get_different_results(const std::vector<T>& cpu_solution,
                           const std::vector<T>& gpu_solution)
    -> std::vector<size_t>
{
    assert(cpu_solution.size() == gpu_solution.size());
    std::vector<size_t> err_indices;
    for (size_t i = 0; i < cpu_solution.size(); i++)
    {
        if (!(cpu_solution[i] == gpu_solution[i]))
            err_indices.emplace_back(i);
    }
    return err_indices;
}

/*
template <typename T, size_t solution_size>
auto get_error_distribution_data(
    const std::array<T, solution_size>& cpu_solution,
    const std::array<T, solution_size>& gpu_solution)
    -> std::tuple<double, double>
{
    std::multiset<double> elements_errors;
    for (auto&& [cpu_val, gpu_val] :
         std::views::zip(cpu_solution, gpu_solution))
    {
        double diff = abs(cpu_val - gpu_val);
        elements_errors.insert(diff);
    }

    double expected_value;
    std::vector<double> error_probability;
    for (auto& err : elements_errors)
    {
        double prob = elements_errors.count(err) / solution_size;
        error_probability.emplace_back(prob);
        expected_value += err * prob;
    }

    double variance;
    for (auto&& [prob, err] :
         std::views::zip(error_probability, elements_errors))
        variance += prob * pow(err - expected_value, 2);

    return {expected_value, variance};
}
*/

csr_matrix::csr_matrix(const char* filename)
{
    int _row_ptr[0], _col_ind[0];
    float _values[0];
    float _matrixDiagonal[0];
    int _num_rows[0];
    int _num_cols[0], _num_vals[0];

    FILE* file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    // Get number of rows, columns, and non-zero value SIGSEGV
    if (fscanf(file, "%d %d %d\n", _num_rows, _num_cols, _num_vals) == EOF)
        printf("Error reading file");

    int* row_ptr_t = (int*)malloc((*_num_rows + 1) * sizeof(int));
    int* col_ind_t = (int*)malloc(*_num_vals * sizeof(int));
    float* values_t = (float*)malloc(*_num_vals * sizeof(float));
    float* matrixDiagonal_t = (float*)malloc(*_num_rows * sizeof(float));
    // Collect occurances of each row for determining the indices of row_ptr
    int* row_occurances = (int*)malloc(*_num_rows * sizeof(int));
    for (int i = 0; i < *_num_rows; i++)
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
    for (int i = 0; i < *_num_rows; i++)
    {
        row_ptr_t[i] = index;
        index += row_occurances[i];
    }
    row_ptr_t[*_num_rows] = *_num_vals;
    free(row_occurances);

    // Set the file position to the beginning of the file
    rewind(file);

    // Read the file again, save column indices and values
    for (int i = 0; i < *_num_vals; i++)
    {
        col_ind_t[i] = -1;
    }

    if (fscanf(file, "%d %d %d\n", _num_rows, _num_cols, _num_vals) == EOF)
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
    row_ptr = row_ptr_t;
    col_ind = col_ind_t;
    values = values_t;
    matrix_diagonal = matrixDiagonal_t;
    num_rows = _num_rows[0];
    num_cols = _num_cols[0];
    num_vals = _num_vals[0];
}

csr_matrix::~csr_matrix()
{
    free(row_ptr);
    free(col_ind);
    free(values);
    free(matrix_diagonal);
}

// CPU implementation of SYMGS using CSR, DO NOT CHANGE THIS
template <typename T>
void symgs_csr_sw(csr_matrix& matrix, std::vector<T>& vector)
{

    // forward sweep
    for (int i = 0; i < matrix.num_rows; i++)
    {
        float sum = vector[i];
        const int row_start = matrix.row_ptr[i];
        const int row_end = matrix.row_ptr[i + 1];
        float currentDiagonal =
            matrix.matrix_diagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= matrix.values[j] * vector[matrix.col_ind[j]];
        }

        sum +=
            vector[i] *
            currentDiagonal; // Remove diagonal contribution from previous loop

        vector[i] = sum / currentDiagonal;
    }

    // backward sweep
    for (int i = matrix.num_rows - 1; i >= 0; i--)
    {
        float sum = vector[i];
        const int row_start = matrix.row_ptr[i];
        const int row_end = matrix.row_ptr[i + 1];
        float currentDiagonal =
            matrix.matrix_diagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= matrix.values[j] * vector[matrix.col_ind[j]];
        }
        sum +=
            vector[i] *
            currentDiagonal; // Remove diagonal contribution from previous loop

        vector[i] = sum / currentDiagonal;
    }
}

// iterate over columns to find unique indices less than row number
auto get_max_iterations(struct csr_matrix& matrix)
{
    std::set<size_t> sf_dependant_idx;
    std::set<size_t> sb_dependant_idx;

    for (int i = 0; i < matrix.num_rows; i++)
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
    return sf_dependant_idx.size();
}

typedef std::chrono::high_resolution_clock::time_point time_var;

template <typename F, typename... Args>
double function_time(F func, Args&&... args)
{
    time_var t1 = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now() - t1)
        .count();
}
