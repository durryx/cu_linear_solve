#pragma once

#include <vector>
struct csr_matrix
{
public:
    int* row_ptr;
    int* col_ind;
    float* values;
    float* matrix_diagonal;
    int num_rows;
    int num_cols;
    int num_vals;

    csr_matrix(const char* filename);

    ~csr_matrix();
};

template <typename T>
void symgs_csr_sw(csr_matrix& matrix, std::vector<T>& vector);

auto get_max_iterations(struct csr_matrix& matrix);
