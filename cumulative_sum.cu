#include "solve.h"
#include <cuda_runtime.h>
#define BLOCK_DIM 256
__global__ void scan_kernel(const float *input,float*output,float *partial_sum,int n)
{
    int tid=threadIdx.x;
    int idx=blockIdx.x*blockDim.x+tid;
    __shared__ float temp1[BLOCK_DIM];
    __shared__ float temp2[BLOCK_DIM];
    float * in_buffer=temp1;
    float * out_buffer=temp2;
    in_buffer[tid]=input[idx];
    __syncthreads();
    for(int d=1;d<=blockDim.x/2;d>>=1)
    {
        if(threadIdx.x>=d)
        {
            out_buffer[tid]=in_buffer[tid]+in_buffer[tid-d];
        }
        else
        {
            out_buffer[tid]=in_buffer[tid];
        }
        __syncthreads();
        float temp=in_buffer;
        in_buffer=out_buffer;
        out_buffer=temp;
    }
    if(threadIdx.x==blockDim.x-1)
    {
        partial_sum[blockIdx.x]=in_buffer[tid];
    }
    output[idx]=in_buffer[tid];
}

__global__ void add_kernel(float*output,float* partial_sum,int n)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(blockIdx.x>0)
    {
        output[idx]+=partial_sum[blockIdx.x-1];
    }
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {
    float *partial_sum_d;
    int threads=BLOCK_DIM;
    int blocks=(N+threads-1)/threads;
    float *partial_sum=new float[blocks];
    cudaMalloc(&partial_sum_d,blocks*sizeof(float));
    scan_kernel<<<blocks,threads>>>(input,output,partial_sum,N);
    cudaDeviceSynchronize();
    cudaMemcpy(partial_sum,partial_sum_d,blocks*sizeof(float),cudaMemcpyDefault);
    for(int i=1;i<blocks;i++)
    {
        partial_sum[i]+=partial_sum[i-1];
    }
    cudaMemcpy(partial_sum_d,partial_sum,blocks*sizeof(float),cudaMemcpyDefault);
    add_kernel<<<blocks,threads>>>(output,partial_sum_d,N);
    cudaDeviceSynchronize();
} 
