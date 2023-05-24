#include "gauss_seidel_sparse.hpp"
#include <iostream>
#include <set>
#include <tuple>
#include <unordered_set>
#include <vector>

csr_matrix::csr_matrix(const char* filename)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    // Get number of rows, columns, and non-zero values
    if (fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals) == EOF)
        printf("Error reading file");

    int* row_ptr_t = (int*)malloc((*num_rows + 1) * sizeof(int));
    int* col_ind_t = (int*)malloc(*num_vals * sizeof(int));
    float* values_t = (float*)malloc(*num_vals * sizeof(float));
    float* matrixDiagonal_t = (float*)malloc(*num_rows * sizeof(float));
    // Collect occurances of each row for determining the indices of row_ptr
    int* row_occurances = (int*)malloc(*num_rows * sizeof(int));
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
}

csr_matrix::~csr_matrix()
{
    free(row_ptr);
    free(col_ind);
    free(values);
    free(matrix_diagonal);
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
