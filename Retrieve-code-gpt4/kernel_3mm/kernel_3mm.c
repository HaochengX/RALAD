void kernel_3mm(int ni,int nj,int nk,int nl,int nm,
                float E[40][50],float A[40][60],float B[60][50],
                float F[50][70],float C[50][80],float D[80][70],
                float G[40][70]) {
    int i, j, k;

#pragma HLS INTERFACE m_axi port=E depth=2000
#pragma HLS INTERFACE m_axi port=A depth=2400
#pragma HLS INTERFACE m_axi port=B depth=3000
#pragma HLS INTERFACE m_axi port=F depth=3500
#pragma HLS INTERFACE m_axi port=C depth=4000
#pragma HLS INTERFACE m_axi port=D depth=5600
#pragma HLS INTERFACE m_axi port=G depth=2800

#pragma HLS ARRAY_PARTITION variable=A cyclic factor=10 dim=2
#pragma HLS ARRAY_PARTITION variable=B cyclic factor=10 dim=1
#pragma HLS ARRAY_PARTITION variable=C cyclic factor=10 dim=2
#pragma HLS ARRAY_PARTITION variable=D cyclic factor=10 dim=1
#pragma HLS ARRAY_PARTITION variable=E complete dim=2
#pragma HLS ARRAY_PARTITION variable=F complete dim=2
#pragma HLS ARRAY_PARTITION variable=G complete dim=2

    // First matrix multiplication E = A * B
    for (i = 0; i < 40; i++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE
        for (j = 0; j < 50; j++) {
            E[i][j] = 0.0;
            for (k = 0; k < 60; k++) {
#pragma HLS UNROLL factor=5
                E[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Second matrix multiplication F = C * D
    for (i = 0; i < 50; i++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE
        for (j = 0; j < 70; j++) {
            F[i][j] = 0.0;
            for (k = 0; k < 80; k++) {
#pragma HLS UNROLL factor=8
                F[i][j] += C[i][k] * D[k][j];
            }
        }
    }

    // Third matrix multiplication G = E * F
    for (i = 0; i < 40; i++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE
        for (j = 0; j < 70; j++) {
            G[i][j] = 0.0;
            for (k = 0; k < 50; k++) {
#pragma HLS UNROLL factor=5
                G[i][j] += E[i][k] * F[k][j];
            }
        }
    }
}