#include "solve.h"
#include <cuda_runtime.h>
#include<iostream>

#define CUDA_CHECK(expr) \
  do { \
    cudaError_t _e = expr; \
    if (_e != cudaSuccess) \
      printf("%s:%d CUDA error %d: %s\n", \
             __FILE__, __LINE__, _e, \
             cudaGetErrorString(_e)); \
  } while(0)

__global__ void blockReduce(const float* in, float* out,
                            int n_samples)
{
  extern __shared__ float sdata[];
  int tid = threadIdx.x,idx = blockIdx.x*blockDim.x + tid;
  float v = (idx < n_samples) ? in[idx] : 0.0f;
  sdata[tid] = v;
  // __syncthreads();

  for(int stride = blockDim.x/2; stride>0; stride>>=1) {
    __syncthreads();
    if(tid < stride)
      sdata[tid] += sdata[tid+stride];
    
  }

  if(tid == 0)
    out[blockIdx.x] = sdata[0];
}

void solve(const float* y_samples,
           float* result, float a, float b,
           int n_samples)
{
  const int THREADS = 256;
  int blocks = (n_samples + THREADS - 1)/THREADS;
  float* block_sum_d,block_sum[blocks],sum[1];
  CUDA_CHECK(cudaMalloc(&block_sum_d,
               blocks * sizeof(float)));
  blockReduce<<<blocks, THREADS,
                THREADS*sizeof(float)>>>(
      y_samples, block_sum_d, n_samples);
  cudaMemcpy(block_sum,block_sum_d,blocks*sizeof(float),cudaMemcpyDeviceToHost);
  cudaFree(block_sum_d);
  sum[0]=0;
  for(int i=0;i<blocks;i++)
  {
    sum[0]+=block_sum[i];
  }
  sum[0]=(b-a)*sum[0]/n_samples;
  cudaMemcpy(result,sum,sizeof(float),cudaMemcpyHostToDevice);
}
