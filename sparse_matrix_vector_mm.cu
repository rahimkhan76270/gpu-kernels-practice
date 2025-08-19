#include <cuda_runtime.h>

__global__ void mat_vec_mult(const float* A, const float* x, float* y, int M, int N)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<M)
    {
        float val=0.0f;
        for(int col=0;col<N;col++)
        {
            val+=A[idx*N+col]*x[col];
        }
        y[idx]=val;
    }
}
__global__ void mat_vec_mult2(const float* A, const float* x, float* y, int M, int N)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int idx=row*N+col;
    if(row<M && col<N && A[idx]!=0)
    {
        atomicAdd(&y[row],A[idx]*x[col]);
    }
}
// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    // int threads=256;
    // int blocks=(M+threads-1)/threads;
    // mat_vec_mult<<<blocks,threads>>>(A,x,y,M,N);
    dim3 threads(16,16);
    dim3 blocks((N+threads.x-1)/threads.x,(M+threads.y-1)/threads.y);
    mat_vec_mult2<<<blocks,threads>>>(A,x,y,M,N);
    cudaDeviceSynchronize();
} 
