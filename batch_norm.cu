#include <cuda_runtime.h>

__global__ void featurewise_mean(const float * input,float *mean,int N,int C)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col<C)
    {
        float sum=0.0f;
        for(int row=0;row<N;row++)
        {
            sum+=input[row*C+col];
        }
        mean[col]=sum/N;
    }
}

__global__ void featurewise_var(const float * input,const float *mean, float * variance,int N,int C)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col<C)
    {
        float sum=0.0f;
        for(int row=0;row<N;row++)
        {
            sum+=(input[row*C+col]-mean[col])*(input[row*C+col]-mean[col]);
        }
        variance[col]=sum/N;
    }
}

__global__ void batch_norm(const float* input, const float* gamma,
                            const float *mean,const float *var, const float* beta, 
                     float* output, int N, int C, float eps)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int idx=row*C+col;
    if(col<C && row<N)
    {
        output[idx]=gamma[col]*((input[idx]-mean[col])/sqrtf(var[col]+eps))+beta[col];
    }
}
// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, 
                     float* output, int N, int C, float eps) {
    float * mean,*var;
    cudaMalloc(&mean,C*sizeof(float));
    cudaMalloc(&var,C*sizeof(float));
    int threads1=256;
    int blocks1=(C+threads1-1)/threads1;
    featurewise_mean<<<blocks1,threads1>>>(input,mean,N,C);
    cudaDeviceSynchronize();
    featurewise_var<<<blocks1,threads1>>>(input,mean,var,N,C);
    cudaDeviceSynchronize();
    dim3 threads2(16,16);
    dim3 blocks2((C+threads2.x-1)/threads2.x,(N+threads2.y-1)/threads2.y);
    batch_norm<<<blocks2,threads2>>>(input,gamma,mean,var,beta,output,N,C,eps);
    cudaDeviceSynchronize();
    cudaFree(mean);
    cudaFree(var);
}
