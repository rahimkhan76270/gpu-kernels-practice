#include <cuda_runtime.h>
// #include <climits.h>
#define BLOCKDIM 256
__global__ void window_sum_kernel(const int *input,int* output,int n,int window_size)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n-window_size+1)
    {
        int val=0;
        for(int i=0;i<window_size;i++)
        {
            val+=input[idx+i];
        }
        output[idx]=val;
    }
}

__global__ void reduce_max(const int * input,int* output,int n)
{
    __shared__ int temp[BLOCKDIM];
    int tid=threadIdx.x;
    int idx=blockIdx.x*blockDim.x+tid;
    temp[tid]=(idx<n)?input[idx]:0;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride=stride/2)
    {
        if(tid<stride)
        {
            temp[tid]=max(temp[tid],temp[tid+stride]);
        }
        __syncthreads();
    }
    if(tid==0)
    {
        atomicMax(output,temp[0]);
    }
    
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    int * window_sum;
    cudaMalloc(&window_sum,(N-window_size+1)*sizeof(int));
    cudaMemset(window_sum,INT_MIN,(N-window_size+1)*sizeof(int));
    int blocks=(N-window_size+BLOCKDIM)/BLOCKDIM;
    window_sum_kernel<<<blocks,BLOCKDIM>>>(input,window_sum,N,window_size);
    cudaDeviceSynchronize();
    reduce_max<<<blocks,BLOCKDIM>>>(window_sum,output,N-window_size+1);
    cudaDeviceSynchronize();
    cudaFree(window_sum);
}
