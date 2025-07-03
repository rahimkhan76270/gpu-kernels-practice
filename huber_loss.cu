#include <cuda_runtime.h>

__device__ float huber_loss(const float x,const float y)
{
    float diff=std::abs(x-y);
    if(diff<1)
    {
        return 0.5*diff*diff;
    }
    else
    {
        return diff-0.5;
    }
}

__global__ void HuberLossKernel(const float* predictions, const float* targets, float* output, size_t n)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n)
    {
        output[idx]=huber_loss(predictions[idx],targets[idx]);
    }
}
// Note: predictions, targets, output are all device pointers to float32 arrays
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {   
    int threads=256;
    int blocks=(n+threads-1)/threads;

    HuberLossKernel<<<blocks,threads>>>(predictions,targets,output,n); 
}
