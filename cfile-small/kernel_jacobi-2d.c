void kernel_jacobi_2d(int tsteps,int n,float A[90][90],float B[90][90])
{
  int t;
  int i;
  int j;
{
    
    
    
    for (t = 0; t < 40; t++) {
      
      
      
      for (i = 1; i < 90 - 1; i++) {
        
        for (j = 1; j < 90 - 1; j++) {
          B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
        }
      }
      
      
      
      for (i = 1; i < 90 - 1; i++) {
        
        for (j = 1; j < 90 - 1; j++) {
          A[i][j] = 0.2 * (B[i][j] + B[i][j - 1] + B[i][1 + j] + B[1 + i][j] + B[i - 1][j]);
        }
      }
    }
  }
}
