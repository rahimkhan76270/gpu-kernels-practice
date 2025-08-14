#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int T=32;

__global__ void gemm_half(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)
{
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int row=by*T+ty;
    int col=bx*T+tx;
    int num_tiles=(K+T-1)/T;

    __shared__ half sh_a[T][T];
    __shared__ half sh_b[T][T];
    float val=0.0f;
    for(int tile=0;tile<num_tiles;tile++)
    {
        //load a;
        if(row<M && tile*T+tx<K)
        {
            sh_a[ty][tx]=A[row*K+tile*T+tx];
        }
        else
        {
            sh_a[ty][tx]=0;
        }
        // load b
        if(col<N && tile*T+ty<K)
        {
            sh_b[ty][tx]=B[(tile*T+ty)*N+col];
        }
        else
        {
            sh_b[ty][tx]=0;
        }
        __syncthreads();
        for(int k_tile=0;k_tile<T;k_tile++)
        {
            val+=static_cast<float>(sh_a[ty][k_tile])*static_cast<float>(sh_b[k_tile][tx]);
        }
        __syncthreads();
    }
    
    if(row<M && col<N)
    {
        C[row*N+col]=static_cast<half>(alpha*val+beta*static_cast<float>(C[row*N+col]));
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 threads(T,T);
    dim3 blocks((N+T-1)/T,(M+T-1)/T);
    gemm_half<<<blocks,threads>>>(A,B,C,M,N,K,alpha,beta);
    cudaDeviceSynchronize();
}
