#include <cuda_runtime.h>


__global__ void ArrSquare(const float *arr,float * out,unsigned int n)
{
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if(idx>=n) return;
    out[idx]=arr[idx]*arr[idx];
}

__global__ void frob_norm_kernel(const float* arr,float * out,const float norm,int size)
{
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if(idx>=size) return;
    out[idx]=arr[idx]/norm;
}
__global__ void sumBlockwise(const float* arr,float * block_sum)
{
    extern __shared__ float temp[];
    int tid=threadIdx.x;
    int idx=blockIdx.x*blockDim.x+tid;
    temp[tid]=arr[idx];
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1)
    {
        if(tid<stride)
        {
            temp[tid]+=temp[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0)
    {
        block_sum[blockIdx.x]=temp[0];
    }
}
float reduce_arr(float* arr,int size)
{
    int threads=256;
    int blocks=(threads+size-1)/threads;
    float *d_in=arr;
    float* d_out;
    cudaMalloc(&d_out,blocks*sizeof(float));
    while(blocks>1)
    {
        sumBlockwise<<<blocks,threads,threads*sizeof(float)>>>(d_in,d_out);
        size=blocks;
        blocks=(size+threads-1)/threads;
        float *temp=d_in;
        d_in=d_out;
        d_out=temp;
    }
    sumBlockwise<<<blocks,threads,threads*sizeof(float)>>>(d_in,d_out);
    float result;
    cudaMemcpy(&result,d_out,sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    return result;
}
// Note: X, Y are all device pointers to float32 arrays
extern "C" void solution(const float* X, float* Y, size_t size) {    
    int threads=256;
    int blocks=(threads+size-1)/threads;
    float *arr_d;
    cudaMalloc(&arr_d,size*sizeof(float));
    ArrSquare<<<blocks,threads>>>(X,arr_d,size);
    cudaDeviceSynchronize();
    float frob_norm=sqrt(reduce_arr(arr_d,size));
    cudaDeviceSynchronize();
    frob_norm_kernel<<<blocks,threads>>>(X,Y,frob_norm,size);
    cudaDeviceSynchronize();
    cudaFree(arr_d);
}
