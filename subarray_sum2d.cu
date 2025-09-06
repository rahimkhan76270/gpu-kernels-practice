#include <cuda_runtime.h>
#define BLOCK_DIM 256
__global__ void reduce_subarray2d(const int * input,int* output,int N,int M,int S_ROW,int E_ROW,int S_COL,int E_COL)
{
    int tid=threadIdx.x;
    int idx=blockIdx.x*blockDim.x+tid;
    int row=idx/M;
    int col=idx%M;
    __shared__ int temp[BLOCK_DIM];
    if(row>=S_ROW && row<=E_ROW && col>=S_COL && col<=E_COL)
    {
        temp[tid]=input[idx];
    }
    else
    {
        temp[tid]=0;
    }
    __syncthreads();
    for(int stride=BLOCK_DIM/2;stride>0;stride/=2)
    {
        if(tid<stride)
        {
            temp[tid]+=temp[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0) atomicAdd(output,temp[0]);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    int threads=BLOCK_DIM;
    int blocks=(N*M+BLOCK_DIM-1)/BLOCK_DIM;
    reduce_subarray2d<<<blocks,threads>>>(input,output,N,M,S_ROW,E_ROW,S_COL,E_COL);
    cudaDeviceSynchronize();
}
