#include<iostream>
#include<cstdlib>
#include<ctime>
#include<cuda_runtime.h>


__global__ void MatVecMult(const float* A,const float *B,float *C,int M,int N)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<M)
    {
        C[i]=0;
        for(int j=blockIdx.y*blockDim.y+threadIdx.y;j<N;j++)
        {
            C[i]+=A[i*M+j]*B[j];
        }
    }
}


int main()
{
    int M=2,N=10;
    srand(time(0));
    int rand_max=10;
    float A[M*N],B[N],C[M];
    float *A_d,*B_d,*C_d;
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            A[i*M+j]=rand()%rand_max;
        }
    }

    for (int i = 0; i < N; i++)
    {
        B[i]=rand()%rand_max;
    }
    
    cudaMalloc(&A_d,M*N*sizeof(float));
    cudaMalloc(&B_d,N*sizeof(float));
    cudaMalloc(&C_d,M*sizeof(float));

    cudaMemcpy(A_d,A,M*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B,N*sizeof(float),cudaMemcpyHostToDevice);
    
    dim3 threads_per_block(16,16);
    dim3 num_blocks((threads_per_block.x+M-1)/threads_per_block.x,(threads_per_block.y+N-1)/threads_per_block.y);

    MatVecMult<<<num_blocks,threads_per_block>>>(A_d,B_d,C_d,M,N);

    cudaMemcpy(C,C_d,M*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            std::cout<<A[i*M+j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    for (int i = 0; i < N; i++)
    {
        std::cout<<B[i]<<" ";
    }

    std::cout<<std::endl;
    for(int i=0;i<M;i++)
    {
        std::cout<<C[i]<<" ";
    }
    std::cout<<std::endl;
    return 0;
}