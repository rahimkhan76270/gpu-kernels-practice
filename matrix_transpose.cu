#include<iostream>
#include<cstdlib>
#include<ctime>
#include<cuda_runtime.h>


__global__ void MatrixTransposeKernel(const float *A,float *B,int M,int N)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;

    if (i<M && j<N)
    {
        B[j*M+i]=A[i*N+j];
    }
    
}

int main()
{
    srand(time(0));
    int rand_max=10;
    int M=10,N=10;
    float A[M*N],B[M*N],*A_d,*B_d;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i*N+j]=rand()%rand_max;
            std::cout<<A[i*N+j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    cudaMalloc(&A_d,M*N*sizeof(float));
    cudaMalloc(&B_d,M*N*sizeof(float));

    cudaMemcpy(A_d,A,M*N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 threads_per_block(16,16);
    dim3 num_blocks((N+threads_per_block.x-1)/threads_per_block.x,(M+threads_per_block.y-1)/threads_per_block.y);

    MatrixTransposeKernel<<<num_blocks,threads_per_block>>>(A_d,B_d,M,N);

    cudaMemcpy(B,B_d,M*N*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            std::cout<<B[i*M+j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    
    return 0;
}