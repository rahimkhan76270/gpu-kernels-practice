#include <cuda_runtime.h>
__global__ void SquareSum(const float* X, float* Y, size_t N, size_t D)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<N)
    {
        Y[row]=0;
        for(int col=0;col<D;col++)
        {
            Y[row]+=X[row*D+col]*X[row*D+col];
        }
        Y[row]=sqrt(Y[row]);
    }
}

__global__ void PairWiseDot(const float* predictions, const float* targets, float* output, size_t n, size_t d)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int idx=row*d+col;
    if(row<n && col<d)
    {
        output[idx]=predictions[idx]*targets[idx];
    }
}

__global__ void RowWiseSum(const float *elememt_wise_dot,float *row_wise_sum,size_t n,size_t d)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<n)
    {
        row_wise_sum[row]=0;
        for(int col=0;col<d;col++)
        {
            row_wise_sum[row]+=elememt_wise_dot[row*d+col];
        }
    }
}

__global__ void CosineSimilarity(const float *dot_prod,const float* square_sum_x,const float* square_sum_y, float * out,int n)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<n)
    {
        out[row]=1-dot_prod[row]/(max(1e-8,square_sum_x[row])*max(1e-8,square_sum_y[row]));
    }
}
// Note: predictions, targets, output are all device pointers to float32 arrays
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n, size_t d) { 
    int threads=256;
    int blocks=(threads+n-1)/threads;
    float *square_sum_x_d,*square_sum_y_d,*pairwise_dot,*row_wise_sum;
    cudaMalloc(&square_sum_x_d,n*sizeof(float));
    cudaMalloc(&square_sum_y_d,n*sizeof(float));
    cudaMalloc(&pairwise_dot,n*d*sizeof(float));
    cudaMalloc(&row_wise_sum,n*sizeof(float));
    SquareSum<<<blocks,threads>>>(predictions,square_sum_x_d,n,d);
    SquareSum<<<blocks,threads>>>(targets,square_sum_y_d,n,d);
    dim3 threads_per_block(16,16);
    dim3 num_blocks((d+15)/16,(n+15/16));
    PairWiseDot<<<num_blocks,threads_per_block>>>(predictions,targets,pairwise_dot,n,d);
    RowWiseSum<<<blocks,threads>>>(pairwise_dot,row_wise_sum,n,d);
    CosineSimilarity<<<blocks,threads>>>(row_wise_sum,square_sum_x_d,square_sum_y_d,output,n);
    cudaFree(square_sum_x_d);
    cudaFree(square_sum_y_d);
    cudaFree(pairwise_dot);
    cudaFree(row_wise_sum);
}
