void kernel_bicg(int m,int n,float A[124][116],float s[116],float q[124],float p[116],float r[124])
{
  int i;
  int j;

  #pragma HLS ARRAY_PARTITION variable=s complete dim=1
  #pragma HLS ARRAY_PARTITION variable=q complete dim=1
  #pragma HLS ARRAY_PARTITION variable=p complete dim=1
  #pragma HLS ARRAY_PARTITION variable=r complete dim=1

  for (i = 0; i < 116; i++) {
    #pragma HLS UNROLL
    s[i] = ((float )0);
  }

  for (I = 0; I < 124; I++) {
    #pragma HLS PIPELINE
    q[I] = 0.0;

    for (j = 0; j < 116; j++) {
      #pragma HLS UNROLL factor=4
      s[j] = s[j] + r[I] * A[I][j];
      q[I] = q[I] + A[I][j] * p[j];
    }
  }
}