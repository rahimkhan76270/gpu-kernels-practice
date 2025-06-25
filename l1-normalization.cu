#include <cuda_runtime.h>

__global__ void RowSum(const float* X, float* Y, size_t B, size_t D)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<B)
    {
        Y[row]=0;
        for(int col=0;col<D;col++)
        {
            Y[row]+=fabsf(X[row*D+col]);
        }
    }
}

__global__ void L1NormalizationKernel(const float* X, float* Y, size_t B, size_t D,const float * row_sum)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int idx=row*D+col;
    if(row<B && col<D)
    {
        Y[idx]=X[idx]/(row_sum[row]+1e-10);
    }
}
extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {    
    float *row_sum_d;
    cudaMalloc(&row_sum_d,B*sizeof(float));
    dim3 threads_per_block(16,16);
    dim3 num_blocks((D+threads_per_block.x-1)/threads_per_block.x,(B+threads_per_block.y-1)/threads_per_block.y);
    int rowBlock = (B + 255) / 256;
    RowSum<<<rowBlock, 256>>>(X, row_sum_d, B, D);
    L1NormalizationKernel<<<num_blocks,threads_per_block>>>(X,Y,B,D,row_sum_d);
    cudaDeviceSynchronize();
    cudaFree(row_sum_d);
}
