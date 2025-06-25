#include <cuda_runtime.h>
#include <math_constants.h> 
#include <float.h>  

__global__ void oneDMaxPoolingKernel(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H,int out_size)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= out_size) return;
    float best = -CUDART_INF_F;

    // compute window base index
    int base = stride*idx - padding;
    for(int m = 0; m < kernel_size; ++m) {
        int in_i = base + dilation*m;
        if(in_i >= 0 && in_i < int(H)) {
            // fmaxf is the float max
            best = fmaxf(best, input[in_i]);
        }
    }
    output[idx] = best;
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H) {    
    dim3 threads_per_block(256);
    dim3 num_blocks((H+threads_per_block.x-1)/threads_per_block.x);
    int eff = int(H) + 2*padding - dilation*(kernel_size - 1);
    int out_size = (eff > 0) ? (eff + stride - 1)/stride : 0;
    oneDMaxPoolingKernel<<<num_blocks,threads_per_block>>>(input,kernel_size,stride,padding,dilation,output,H,out_size);
    cudaDeviceSynchronize();
}
