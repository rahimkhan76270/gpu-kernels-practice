#include <cuda_runtime.h>

// Note: input, output are all device pointers to float32 arrays
__global__ void SELUKernel(const float* input, float* output, size_t n, size_t m)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int idx=row*n+col;
    if(row<m && col<n)
    {
        output[idx]=1.0507*(max((float)0,(float)input[idx])+min((float)0,(float)1.67326*(exp(input[idx])-1)));
    }
}
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {    
    dim3 threds_per_block(16,16);
    dim3 num_blocks((n+threds_per_block.x-1)/threds_per_block.x,(m+threds_per_block.y-1)/threds_per_block.y);
    SELUKernel<<<num_blocks,threds_per_block>>>(input,output,n,m);
}
