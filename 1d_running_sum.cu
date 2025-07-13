#include <cuda_runtime.h>

__global__ void RunningSumKernel(const float* input, int W, float* output, size_t N)
{
    extern __shared__ float temp[];
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    temp[threadIdx.x]=(idx<N)?input[idx]:0.0f;
    __syncthreads();
    if(idx<N)
    {
        float val=0;
        for(int j=-1*(W/2);j<W/2 +1;j++)
        {
            if(idx+j>=0 && idx+j<N) val+=temp[threadIdx.x+j];
            __syncthreads();
        }
        output[idx]=val;
    }
}
// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, size_t W, float* output, size_t N) {    
    int threads=256;
    int blocks=(N+threads-1)/threads;
    RunningSumKernel<<<blocks,threads,threads*sizeof(float)>>>(input,W,output,N);
    cudaDeviceSynchronize();
}
