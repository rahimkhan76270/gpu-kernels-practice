#include <cuda_runtime.h>

__global__ void DiagonalMatMul(const float* diagonal_a, const float* input_b, float* output_c, size_t n, size_t m)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    int col=blockIdx.y*blockDim.y+threadIdx.y;
    if(row<n && col<m)
    {
        output_c[row*m+col]=diagonal_a[row]*input_b[row*m+col];
    }
}
// Note: diagonal_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* diagonal_a, const float* input_b, float* output_c, size_t n, size_t m) { 
    dim3 threads(16,16);
    dim3 blocks((n+15)/16,(m+15)/16);
    DiagonalMatMul<<<blocks,threads>>>(diagonal_a,input_b,output_c,n,m);
    cudaDeviceSynchronize();   
}
