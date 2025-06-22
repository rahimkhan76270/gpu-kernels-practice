#include <cuda_runtime.h>
#include <cmath>

__global__ void GELUKernel(const float* input, float* output, size_t n, size_t m)
{
    int col=blockIdx.y*blockDim.y+threadIdx.y;
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    int idx=row*n+col;
    if(row<m && col<n)
    {
        output[idx]=0.5*input[idx]*(1+tanh(sqrt(2/M_PI)*(input[idx]+0.044715*pow(input[idx],3))));
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {    
    dim3 threads_per_block(16,16);
    dim3 num_blocks((m+threads_per_block.x-1)/threads_per_block.x,(n+threads_per_block.y-1)/threads_per_block.y);
    GELUKernel<<<num_blocks,threads_per_block>>>(input,output,n,m);
}
