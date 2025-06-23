#include <cuda_runtime.h>
#include<cmath>

__global__ void EluKernel(const float* input, float* output, size_t n, size_t m, float alpha)
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
            output[idx]=alpha*(exp(input[idx])-1);
        }
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {    
    dim3 threads_per_block(16,16);
    dim3 num_blocks((n+threads_per_block.x-1)/threads_per_block.x,(m+threads_per_block.y-1)/threads_per_block.y);
    EluKernel<<<num_blocks,threads_per_block>>>(input,output,n,m,alpha);
    // cudaDeviceSynchronize();
}
