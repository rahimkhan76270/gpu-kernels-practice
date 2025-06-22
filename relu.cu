#include <cuda_runtime.h>


__global__ void ReLUKernel(const float* input, float* output, size_t n, size_t m)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int idx=row*n+col;
    if(row<m && col<n)
    {
        if(input[idx]>0)
        {
            output[idx]=input[idx];
        }
        else
        {
            output[idx]=0;
        }
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {    
    dim3 thread_per_block(16,16);
    dim3 num_blocks((n+thread_per_block.x-1)/thread_per_block.x,(m+thread_per_block.y-1)/thread_per_block.y);
    ReLUKernel<<<num_blocks,thread_per_block>>>(input,output,n,m);
}
