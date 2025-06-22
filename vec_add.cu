#include<iostream>
#include<cuda_runtime.h>
#include<ctime>
#include<cstdlib>

__global__ void vec_addition(float*a , float*b, float* c,int vec_size)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<vec_size)
    {
        c[i]=a[i]+b[i];
    }
}


int main()
{
    int vec_size=10000;
    float a[vec_size],b[vec_size],c[vec_size];
    int rand_max=100;
    srand(time(0));
    for(int i=0;i<vec_size;i++)
    {
        a[i]=rand()%rand_max;
        b[i]=rand()%rand_max;
    }

    float* a_d,*b_d,*c_d;
    cudaMalloc(&a_d,vec_size*sizeof(float));
    cudaMalloc(&b_d,vec_size*sizeof(float));
    cudaMalloc(&c_d,vec_size*sizeof(float));
    cudaMemcpy(a_d,a,vec_size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,vec_size*sizeof(float),cudaMemcpyHostToDevice);
    int threads_per_block=16;
    int block=(vec_size+threads_per_block-1)/threads_per_block;
    vec_addition<<<block,threads_per_block>>>(a_d,b_d,c_d,vec_size);
    cudaMemcpy(c,c_d,vec_size*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    for(auto x:a)std::cout<<x<<" ";
    std::cout<<"\n";
    for(auto x:b) std::cout<<x<<" ";
    std::cout<<"\n";
    for(auto x:c) std::cout<<x<<" ";
    std::cout<<"\n";
    return 0;
}