#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>


__global__ void mat_mult(const float *A,const float *B,float* C,int M,int N,int K)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    if (i<M && j<K)
    {
        C[i*M+j]=0;
        for(int k=0;k<N;k++)
        {
            C[i*M+j]+=A[i*M+k]*B[k*N+j];
        }
    }
    
}


int main()
{
    int M = 3, N = 3, K = 3;
    srand(time(0));
    int    rand_max = 10;
    float  A[M * N], B[N * K], C[M * K];
    float *A_d, *B_d, *C_d;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * M + j] = rand() % rand_max;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            B[i * N + j] = rand() % rand_max;
        }
    }

    cudaMalloc(&A_d, M * N * sizeof(float));
    cudaMalloc(&B_d, N * K * sizeof(float));
    cudaMalloc(&C_d,M*K*sizeof(float));

    cudaMemcpy(A_d, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads_per_block(16,16);
    dim3 num_blocks((threads_per_block.x+M-1)/threads_per_block.x,(threads_per_block.y+N-1)/threads_per_block.y);

    mat_mult<<<num_blocks,threads_per_block>>>(A_d,B_d,C_d,M,N,K);

    cudaMemcpy(C,C_d,M*K*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout<<A[i * M + j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
           std::cout<<B[i * N + j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
           std::cout<<C[i * N + j]<<" ";
        }
        std::cout<<std::endl;
    }
    return 0;
}