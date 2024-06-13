void kernel_atax(int m, int n, float A[116][124], float x[124], float y[124], float tmp[116]) {
  int i;
  int j;

  // Initialize y array
  for (i = 0; i < 124; i++) {
    #pragma HLS UNROLL
    y[i] = 0.0;
  }

  // Main computation
  for (i = 0; i < 116; i++) {
    #pragma HLS LOOP UNROLL factor=2
    tmp[i] = 0.0;

    for (j = 0; j < 124; j++) {
      #pragma HLS PIPELINE II=1
      tmp[i] += A[i][j] * x[j];
    }

    for (j = 0; j < 124; j++) {
      #pragma HLS PIPELINE II=1
      y[j] += A[i][j] * tmp[i];
    }
  }
}