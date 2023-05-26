#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#define THREADN 320
#define CHUNKSN 10
#define LINESPERTHREAD 8

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t err = call;                                          \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, \
                   __LINE__);                                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define CHECK_KERNELCALL()                                                     \
    {                                                                          \
        const cudaError_t err = cudaGetLastError();                            \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, \
                   __LINE__);                                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

double get_time()
{ // function to get the time of day in second
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row)
// format
void read_matrix(int** row_ptr, int** col_ind, float** values,
                 float** matrixDiagonal, const char* filename, int* num_rows,
                 int* num_cols, int* num_vals)
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
    *row_ptr = row_ptr_t;
    *col_ind = col_ind_t;
    *values = values_t;
    *matrixDiagonal = matrixDiagonal_t;
}

// CPU implementation of SYMGS using CSR, DO NOT CHANGE THIS
void symgs_csr_sw(const int* row_ptr, const int* col_ind, const float* values,
                  const int num_rows, float* x, float* matrixDiagonal)
{

    // forward sweep
    for (int i = 0; i < num_rows; i++)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }

        sum +=
            x[i] *
            currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }

    // backward sweep
    for (int i = num_rows - 1; i >= 0; i--)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }
        sum +=
            x[i] *
            currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }
}

// GPU implementation of SYMGS using CSR
__global__ void symgs_csr_gpu_forward(const int* row_ptr, const int* col_ind,
                                      const float* values, const int num_rows,
                                      float* x, float* matrixDiagonal,
                                      float* x2, char* locks, char* not_done,
                                      char* thread_done, int index0)
{
    int start, end, i;
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    index += index0;

    if (thread_done[index])
        return;
    thread_done[index] = 1;

    start = LINESPERTHREAD * index;
    end = LINESPERTHREAD * (index + 1);

    if (start >= num_rows)
        return;

    if (end > num_rows)
        end = num_rows;

    for (i = start; i < end; i++)
    {
        // if I alread calculated the value for this row: continue
        if (locks[i])
            continue;

        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i];

        char missed = 0;
        for (int j = row_start; j < row_end; j++)
        {
            int col_index = col_ind[j];

            // needed to address rogue -1 col_index
            if (col_index < 0)
                continue;

            // if I need a new value
            if (col_index < i)
            {
                // if that value is not ready yet: skip this round, try next
                // time
                if (locks[col_index] == 0)
                {
                    *not_done = 1;
                    thread_done[index] = 0;
                    missed = 1;
                    break;
                }
                // else take the new value from x2 - new values array
                sum -= (float)(((double)values[j]) * ((double)x2[col_index]));
            }
            // else take old value
            else
            {
                sum -= (float)(((double)values[j]) * ((double)x[col_index]));
            }
        }
        // if I had to break from previous loop: skip setting new value
        if (missed)
            continue;

        sum += (float)(((double)x[i]) * ((double)currentDiagonal));
        x2[i] = (float)(((double)sum) / ((double)currentDiagonal));
        locks[i] = 1;
    }
}

__global__ void symgs_csr_gpu_backward(const int* row_ptr, const int* col_ind,
                                       const float* values, const int num_rows,
                                       float* x, float* matrixDiagonal,
                                       float* x2, char* locks, char* not_done,
                                       char* thread_done, int index0)
{
    int start, end, i;
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    index += index0;

    if (thread_done[index] == 2)
        return;
    thread_done[index] = 2;

    start = LINESPERTHREAD * index;
    end = LINESPERTHREAD * (index + 1);

    if (start >= num_rows)
        return;

    if (end > num_rows)
        end = num_rows;

    // Now x becomes the new value array and x2 becomes the old value array
    for (i = end - 1; i >= start; i--)
    {
        // if I alread calculated the value for this row: continue
        if (locks[i] >> 1)
            continue;

        float sum = x2[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i];

        char missed = 0;
        for (int j = row_start; j < row_end; j++)
        {
            int col_index = col_ind[j];

            // needed to address rogue -1 col_index
            if (col_index < 0)
                continue;

            // if I need a new value
            if (col_index > i)
            {
                // new value is not ready yet, try next iteration
                if (locks[col_index] < 2)
                {
                    *not_done = 1;
                    thread_done[i] = 1;
                    missed = 1;
                    break;
                }
                // else take the new value from x - new values array
                sum -= (float)((double)values[j] * (double)x[col_index]);
            }
            else
            {
                // else take old value
                sum -= (float)((double)values[j] * (double)x2[col_index]);
            }
        }
        // if I had to break from previous loop: skip setting new value
        if (missed)
            continue;

        sum += (float)((double)x2[i] * (double)currentDiagonal);
        x[i] = (float)((double)sum / (double)currentDiagonal);
        locks[i] = 2;
    }
}

int main(int argc, const char* argv[])
{
    if (argc != 2)
    {
        printf("Usage: ./exec matrix_file");
        return 0;
    }

    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float* values;
    float* matrixDiagonal;

    const char* filename = argv[1];

    double start_cpu, end_cpu;
    double start_gpu, end_gpu;

    read_matrix(&row_ptr, &col_ind, &values, &matrixDiagonal, filename,
                &num_rows, &num_cols, &num_vals);
    float* x = (float*)malloc(num_rows * sizeof(float));
    float* xCopy = (float*)malloc(num_rows * sizeof(float));

    // Generate a random vector
    srand(time(NULL));
    for (int i = 0; i < num_rows; i++)
    {
        x[i] = (float)(rand() % 100) /
               (float)(rand() % 100 + 1); // the number we use to divide cannot
                                          // be 0, that's the reason of the +1
        xCopy[i] = x[i];
    }

    // ############################ cpu part ############################
    start_cpu = get_time();
    printf("rows: %d\n", num_rows);
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_cpu = get_time();

    // ############################ gpu part ############################

    int blockN = num_rows / (THREADN * LINESPERTHREAD) + 1;

    // allocate space
    int *dev_row_ptr, *dev_col_ind;
    float *dev_values, *dev_x, *dev_matrixDiagonal, *dev_x2;
    char *dev_semaphores, *dev_not_done, *dev_thread_done;
    CHECK(cudaMalloc(&dev_row_ptr, (num_rows + 1) * sizeof(int)));
    CHECK(cudaMalloc(&dev_col_ind, num_vals * sizeof(int)));
    CHECK(cudaMalloc(&dev_values, num_vals * sizeof(float)));
    CHECK(cudaMalloc(&dev_x, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&dev_matrixDiagonal, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&dev_x2, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&dev_semaphores, num_rows * sizeof(char)));
    CHECK(cudaMalloc(&dev_not_done, sizeof(char)));
    CHECK(cudaMalloc(&dev_thread_done, (blockN * THREADN) * sizeof(char)));

    CHECK(cudaMemcpy(dev_row_ptr, row_ptr, (num_rows + 1) * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_col_ind, col_ind, num_vals * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_values, values, num_vals * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_x, xCopy, num_rows * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_matrixDiagonal, matrixDiagonal,
                     num_rows * sizeof(float), cudaMemcpyHostToDevice));

    // initialize lock and done vectors to all 0 in host then copy to device
    char* host_semaphores = (char*)calloc(num_rows, sizeof(char));
    char* host_thread_done = (char*)calloc((blockN * THREADN), sizeof(char));
    CHECK(cudaMemcpy(dev_semaphores, host_semaphores, num_rows * sizeof(char),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_thread_done, host_thread_done,
                     (blockN * THREADN) * sizeof(char),
                     cudaMemcpyHostToDevice));

    // compute in gpu
    start_gpu = get_time();

    dim3 threadsPerBlock(THREADN, 1, 1);

    // divide forward and backward sweep into chunks, rerun each chunk until all
    // values have been found forward sweep
    int blocksToCompute = blockN;
    for (int i = 0; i < CHUNKSN; i++)
    {
        int blocksInRound = blocksToCompute / (CHUNKSN - i);
        dim3 blocksPerGrid(blocksInRound, 1, 1);
        char not_done;
        do
        {
            // reset not done to 0 in gpu for every cycle
            not_done = 0;
            CHECK(cudaMemcpy(dev_not_done, &not_done, sizeof(char),
                             cudaMemcpyHostToDevice));

            symgs_csr_gpu_forward<<<blocksPerGrid, threadsPerBlock>>>(
                dev_row_ptr, dev_col_ind, dev_values, num_rows, dev_x,
                dev_matrixDiagonal, dev_x2, dev_semaphores, dev_not_done,
                dev_thread_done, (blockN - blocksToCompute) * THREADN);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(&not_done, dev_not_done, sizeof(char),
                             cudaMemcpyDeviceToHost));
        } while (not_done);

        blocksToCompute -= blocksInRound;
    }

    // backward sweep
    blocksToCompute = blockN;
    for (int i = 0; i < CHUNKSN; i++)
    {
        int blocksInRound = blocksToCompute / (CHUNKSN - i);
        dim3 blocksPerGrid(blocksInRound, 1, 1);
        blocksToCompute -= blocksInRound;

        char not_done;
        do
        {
            not_done = 0;
            CHECK(cudaMemcpy(dev_not_done, &not_done, sizeof(char),
                             cudaMemcpyHostToDevice));

            symgs_csr_gpu_backward<<<blocksPerGrid, threadsPerBlock>>>(
                dev_row_ptr, dev_col_ind, dev_values, num_rows, dev_x,
                dev_matrixDiagonal, dev_x2, dev_semaphores, dev_not_done,
                dev_thread_done, (blocksToCompute)*THREADN);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(&not_done, dev_not_done, sizeof(char),
                             cudaMemcpyDeviceToHost));
        } while (not_done);
    }

    end_gpu = get_time();

    CHECK(cudaMemcpy(xCopy, dev_x, num_rows * sizeof(float),
                     cudaMemcpyDeviceToHost));

    // check for errors, sensibility is needed because gpu values slightly
    // differ
    int errors = 0;
    float maxError = 0.0;
    for (int i = 0; i < num_rows; i++)
    {
        if ((x[i] - xCopy[i] > 0.0001 || x[i] - xCopy[i] < -0.0001) &&
            (x[i] - xCopy[i]) / x[i] > 0.001)
        {

            float err = x[i] - xCopy[i];
            err = err > 0 ? err : -err;
            maxError = err > maxError ? err : maxError;

            errors++;
        }
    }

    if (errors > 0)
        printf("Errors: %d\nMax error: %lf\n", errors, maxError);

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
    free(host_semaphores);
    free(host_thread_done);

    CHECK(cudaFree(dev_row_ptr));
    CHECK(cudaFree(dev_col_ind));
    CHECK(cudaFree(dev_values));
    CHECK(cudaFree(dev_x));
    CHECK(cudaFree(dev_matrixDiagonal));
    CHECK(cudaFree(dev_x2));
    CHECK(cudaFree(dev_semaphores));
    CHECK(cudaFree(dev_thread_done));
    CHECK(cudaFree(dev_not_done));

    return 0;
}