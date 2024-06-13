#include <math.h>

// Assume max size to be 120 for the matrix
#define MAX_SIZE 120

void kernel_cholesky(int n, float A[MAX_SIZE][MAX_SIZE])
{
    int i, j, k;

    // Pragma to improve memory access patterns and pipeline efficiency
    #pragma HLS ARRAY_PARTITION variable=A cyclic factor=30 dim=2

    for (i = 0; i < n; i++) {
        // Inner loop pipelining for parallel execution
        #pragma HLS PIPELINE

        for (j = 0; j < i; j++) {
            float sum = A[i][j];

            // Fully unroll this loop for maximum parallelism
            #pragma HLS UNROLL factor=30
            for (k = 0; k < j; k++) {
                sum -= A[i][k] * A[j][k];
            }

            A[i][j] = sum / A[j][j];
        }

        float sum = A[i][i];

        // Fully unroll this loop to remove dependency and increase parallelism
        #pragma HLS UNROLL factor=30
        for (k = 0; k < i; k++) {
            sum -= A[i][k] * A[i][k];
        }

        A[i][i] = sqrt(sum);
    }
}