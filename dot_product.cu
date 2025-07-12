#include "solve.h"
#include <cuda_runtime.h>

__global__ void ElementMul(const float* A, const float* B, float* result, int N)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N)
    {
        result[idx]=A[idx]*B[idx];
    }
}

__global__ void reduce(const float *element_mul,float *result,int N)
{
    extern __shared__ float temp[];
    int tid=threadIdx.x;
    int idx=blockIdx.x*blockDim.x+tid;
    temp[tid]=(idx<N)?element_mul[idx]:0.0f;
    __syncthreads();

    for(int d=blockDim.x/2;d>0;d>>=1)
    {
        if(tid<d)
        {
            temp[tid]+=temp[tid+d];
        }
        __syncthreads();
    }
    if(tid==0)
    {
        atomicAdd(result,temp[0]);
    }
}
// A, B, result are device pointers
void solve(const float* A, const float* B, float* result, int N) {
    float* element_mul;
    cudaMalloc(&element_mul,N*sizeof(float));
    int threads=256;
    int blocks=(N+threads-1)/threads;
    ElementMul<<<blocks,threads>>>(A,B,element_mul,N);
    cudaDeviceSynchronize();
    reduce<<<blocks,threads,threads*sizeof(float)>>>(element_mul,result,N);
    cudaDeviceSynchronize();
    cudaFree(element_mul);
}
