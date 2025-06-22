#include <cuda_runtime.h>

__global__ void AvgPooling1D(const float* input, int kernel_size, int stride, int padding, float* output, size_t H,int out_size)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<out_size)
    {
        output[i]=0;
        for(int m=0;m<kernel_size;m++)
        {
            if(stride*i+m-padding<H) output[i]+=input[stride*i+m-padding];
        }
        output[i]=output[i]/kernel_size;
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    dim3 thread_per_block(16);
    dim3 num_blocks((thread_per_block.x+H-1)/thread_per_block.x);
    int out_size=floor((H+2*padding-kernel_size)/stride +1);
    AvgPooling1D<<<num_blocks,thread_per_block>>>(input,kernel_size,stride,padding,output, H,out_size);
}