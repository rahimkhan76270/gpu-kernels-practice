#include <cuda_runtime.h>

__device__ float hinge_loss(const float x,const float y)
{
    return max((float)0,(float)(1-x*y));
}

__global__ void HingeLossKernel(const float* predictions, const float* targets, float* output, size_t n)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n)
    {
        output[idx]=hinge_loss(predictions[idx],targets[idx]);
    }
}
// Note: predictions, targets, output are all device pointers to float32 arrays
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {   
    int threads=256;
    int blocks=(n+threads-1)/threads;
    HingeLossKernel<<<blocks,threads>>>(predictions,targets,output,n); 
}
