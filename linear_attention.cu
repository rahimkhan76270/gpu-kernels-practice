#include <cuda_runtime.h>

__device__ float Phi(float x)
{
    if(x>0) return x+1;
    return __expf(x);
}

__global__  void sigma_k(const float* K, float* output, int M, int d)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if (col<d)
    {
        float val=0.0f;
        for (size_t row = 0; row < M; row++)
        {
            val+=Phi(K[row*d+col]); 
        }
        output[col]=val;
    }
}

__global__ void denominator(const float* Q, const float* sigmak, float* output, int M, int d)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<M)
    {
        float val=0.0f;
        for (int col = 0; col < d; col++)
        {
            val+=Phi(Q[row*d+col])*sigmak[col];
        }
        output[row]=val;
    }
}
__global__ void KTV(const float* K, const float* V, float* output, int M, int d)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<d && col<d)
    {
        float val=0.0f;
        for (size_t i = 0; i < M; i++)
        {
            val+=Phi(K[i*d+row])*V[i*d+col];
        }
        output[row*d+col]=val;
    }
}

__global__ void LinearSelfAttention(const float* Q, const float* Ktv, const float* denom, float* output, int M, int d)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<M && col<d)
    {
        float val=0.0f;
        for (size_t i = 0; i < d; i++)
        {
            val+=Phi(Q[row*d+i])*Ktv[i*d+col];
        }
        output[row*d+col]=val/denom[row];
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    float * sigmk,*ktv,*denom;
    cudaMalloc(&sigmk,sizeof(float)*d);
    cudaMalloc(&ktv,d*d*sizeof(float));
    cudaMalloc(&denom,M*sizeof(float));
    int threads1=256;
    int blocks1=(d+threads1-1)/threads1;
    sigma_k<<<blocks1,threads1>>>(K,sigmk,M,d);
    cudaDeviceSynchronize();
    int blocks2=(M+threads1-1)/threads1;
    denominator<<<blocks2,threads1>>>(Q,sigmk,denom,M,d);
    cudaDeviceSynchronize();
    dim3 threads3(16,16);
    dim3 blocks3((d+threads3.x-1)/threads3.x,(d+threads3.y-1)/threads3.y);
    KTV<<<blocks3,threads3>>>(K,V,ktv,M,d);
    cudaDeviceSynchronize();
    dim3 blocks4((d+threads3.x-1)/threads3.x,(M+threads3.y-1)/threads3.y);
    LinearSelfAttention<<<blocks4,threads3>>>(Q,ktv,denom,output,M,d);
    cudaDeviceSynchronize();
    cudaFree(sigmk);
    cudaFree(ktv);
    cudaFree(denom);
}
