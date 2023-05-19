
void read_matrix(int** row_ptr, int** col_ind, float** values,
                 float** matrixDiagonal, const char* filename, int* num_rows,
                 int* num_cols, int* num_vals);

void symgs_csr_sw(const int* row_ptr, const int* col_ind, const float* values,
                  const int num_rows, float* x, float* matrixDiagonal);

void gauss_seidel_sparse_solve();

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

    virtual ~csr_matrix();
};
