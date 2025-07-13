#include "solve.h"
#include <cuda_runtime.h>


__global__ void CatLossKernel(const float* logits, const int* true_labels, float* loss_arr, int N, int C)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N)
    {
        float loss=0.0f;
        for(int col=0;col<C;col++)
        {
            loss+=__expf(logits[idx*C+col]);
        }

        atomicAdd(loss_arr,(__logf(loss)-logits[idx*C+true_labels[idx]])/N);
    }
}
// logits, true_labels, loss are device pointers
void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    int threads=256;
    int blocks=(N+threads-1)/threads;
    CatLossKernel<<<blocks,threads>>>(logits,true_labels,loss,N,C);
    cudaDeviceSynchronize();
}
