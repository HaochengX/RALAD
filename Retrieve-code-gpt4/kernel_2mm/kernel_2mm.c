void kernel_2mm(int ni, int nj, int nk, int nl, float alpha, float beta, float tmp[40][50], float A[40][70], float B[70][50], float C[50][80], float D[40][80]) {
    int i, j, k;

    // Interface pragmas for external memory interfaces
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem4

    // Dataflow to allow parallel execution of the two major loop nests
#pragma HLS dataflow

    // First major loop nest: Matrix multiplication for tmp
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
            tmp[i][j] = 0.0;
            for (k = 0; k < nk; ++k) {
                tmp[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }

    // Second major loop nest: Matrix multiplication and scaling for D
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
#pragma HLS PIPELINE II=1
            D[i][j] *= beta;
            for (k = 0; k < nj; ++k) {
                D[i][j] += tmp[i][k] * C[k][j];
            }
        }
    }
}