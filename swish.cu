#include <cuda_runtime.h>

__device__ float swish(const float x)
{
    float sigmd=(float)1/(float)(1+exp(-x));
    return x*sigmd;
}

__global__ void SwishKernel(const float* input, float* output, size_t n, size_t m)
{
    int col=blockIdx.y*blockDim.y+threadIdx.y;
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    int idx=row*n+col;
    if(row<m && col<n)
    {
        output[idx]=swish(input[idx]);
    }
}
// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {    
    dim3 threads(16,16);
    dim3 blocks((m+15)/16,(n+15)/16);
    SwishKernel<<<blocks,threads>>>(input,output,n,m);
}
