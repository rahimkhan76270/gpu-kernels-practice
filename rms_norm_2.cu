#include <cuda_runtime.h>
#define BLOCK_DIM 256
__global__ void reduce(const float * input,float *output,int n,float eps)
{
    __shared__ float temp[BLOCK_DIM];
    int tid=threadIdx.x;
    int idx=blockIdx.x*BLOCK_DIM+tid;
    temp[tid]=(idx<n)?(input[idx])*(input[idx]):0.0f;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride=stride/2)
    {
        if(tid<stride)
        {
            temp[tid]+=temp[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0)atomicAdd(output,(temp[0]+eps)/n);
}

__global__ void rms_val(const float *input,float *output,float *rms_sqrd,int n,float gamma,float beta)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n)
    {
        output[idx]=(input[idx]/sqrt(*rms_sqrd))*gamma +beta;
    }
}
// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, 
                     float* output, int N, float eps) {
    float *rms;
    cudaMalloc(&rms,sizeof(float));
    int blocks=(BLOCK_DIM+N-1)/BLOCK_DIM;
    reduce<<<blocks,BLOCK_DIM>>>(input,rms,N,eps);
    cudaDeviceSynchronize();

    rms_val<<<blocks,BLOCK_DIM>>>(input,output,rms,N,gamma,beta);
    cudaDeviceSynchronize();
    cudaFree(rms);
}
