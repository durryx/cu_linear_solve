#pragma once

template <typename T>
void gauss_seidel_sparse_solve(csr_matrix& matrix, std::vector<T>& vector,
                               int device);
