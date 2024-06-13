void kernel_adi(int tsteps, int n, float u[60][60], float v[60][60], float p[60][60], float q[60][60])
{
    float DX, DY, DT, B1, B2, mul1, mul2, a, b, c, d, e, f;
    int t, i, j;
    
    DX = 1.0 / 60.0;
    DY = 1.0 / 60.0;
    DT = 1.0 / 40.0;
    B1 = 2.0;
    B2 = 1.0;
    mul1 = B1 * DT / (DX * DX);
    mul2 = B2 * DT / (DY * DY);
    a = -mul1 / 2.0;
    b = 1.0 + mul1;
    c = a;
    d = -mul2 / 2.0;
    e = 1.0 + mul2;
    f = d;

    #pragma HLS dataflow

    for (t = 1; t <= tsteps; t++) {
        // Column Sweep
        for (i = 1; i < n - 1; i++) {
            #pragma HLS loop_tripcount min=58 max=58
            v[0][i] = 1.0;
            p[i][0] = 0.0;
            q[i][0] = v[0][i];
            
            for (j = 1; j < n - 1; j++) {
                #pragma HLS pipeline
                p[i][j] = -c / (a * p[i][j - 1] + b);
                q[i][j] = (-d * u[j][i - 1] + (1.0 + 2.0 * d) * u[j][i] - f * u[j][i + 1] - a * q[i][j - 1]) / (a * p[i][j - 1] + b);
            }
            v[n - 1][i] = 1.0;
            
            for (j = n - 2; j >= 0; j--) {
                #pragma HLS loop_tripcount min=58 max=58
                int _in_j_0 = j + 1;
                v[j][i] = p[i][j] * v[_in_j_0][i] + q[i][j];
            }
        }
        // Row Sweep
        for (i = 1; i < n - 1; i++) {
            #pragma HLS loop_tripcount min=58 max=58
            u[i][0] = 1.0;
            p[i][0] = 0.0;
            q[i][0] = u[i][0];
            
            for (j = 1; j < n - 1; j++) {
                #pragma HLS pipeline
                p[i][j] = -f / (d * p[i][j - 1] + e);
                q[i][j] = (-a * v[i - 1][j] + (1.0 + 2.0 * a) * v[i][j] - c * v[i + 1][j] - d * q[i][j - 1]) / (d * p[i][j - 1] + e);
            }
            u[i][n - 1] = 1.0;
            
            for (j = n - 2; j >= 0; j--) {
                #pragma HLS loop_tripcount min=58 max=58
                int _in_j = j + 1;
                u[i][j] = p[i][j] * u[i][_in_j] + q[i][j];
            }
        }
    }
}