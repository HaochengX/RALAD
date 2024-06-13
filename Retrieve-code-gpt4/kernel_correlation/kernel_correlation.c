#include <math.h>

// Declare the kernel function to be synthesized
void kernel_correlation(int m, int n, float float_n, float data[100][80], float corr[80][80], float mean[80], float stddev[80])
{
  float eps = 0.1;
  int i, j, k;

  // HLS directives for array partitioning to allow parallel access
  #pragma HLS ARRAY_PARTITION variable=mean complete dim=1
  #pragma HLS ARRAY_PARTITION variable=stddev complete dim=1

  // Compute mean
  for (j = 0; j < 80; j++) {
    #pragma HLS LOOP_UNROLL
    mean[j] = 0.0;
    
    for (i = 0; i < 100; i++) {
      #pragma HLS PIPELINE
      mean[j] += data[i][j];
    }
    mean[j] /= float_n;
  }

  // Compute standard deviation
  for (j = 0; j < 80; j++) {
    #pragma HLS LOOP_UNROLL
    stddev[j] = 0.0;
    
    for (i = 0; i < 100; i++) {
      #pragma HLS PIPELINE
      stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
    }
    stddev[j] /= float_n;
    stddev[j] = sqrt(stddev[j]);
    // Handling near-zero standard deviations
    stddev[j] = (stddev[j] <= eps ? 1.0 : stddev[j]);
  }

  // Normalize data
  for (i = 0; i < 100; i++) {
    for (j = 0; j < 80; j++) {
      #pragma HLS PIPELINE
      data[i][j] -= mean[j];
      data[i][j] /= sqrt(float_n) * stddev[j];
    }
  }

  // Compute correlation matrix
  for (i = 0; i < 80 - 1; i++) {
    corr[i][i] = 1.0;
    
    for (j = i + 1; j < 80; j++) {
      #pragma HLS LOOP_FLATTEN off
      corr[i][j] = 0.0;
      
      for (k = 0; k < 100; k++) {
        #pragma HLS PIPELINE
        corr[i][j] += data[k][i] * data[k][j];
      }
      corr[j][i] = corr[i][j];
    }
  }
  corr[80 - 1][80 - 1] = 1.0;
}