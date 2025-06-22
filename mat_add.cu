#include<iostream>
#include<cuda_runtime.h>
#include<cstdlib>
#include<ctime>

__global__ void mat_add(float*a,float*b,float*c,int m,int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i<m && j<n)
    {
        int idx=i*n+j;
        c[idx]=a[idx]+b[idx];
    }
}


int main(){
    srand(time(0));
    int rand_max=10;
    float *a,*b,*c,*a_d,*b_d,*c_d;
    int m=3,n=3;
    a=(float*)malloc(m*n*sizeof(float));
    b=(float*)malloc(m*n*sizeof(float));
    c=(float*)malloc(m*n*sizeof(float));
    cudaMalloc(&a_d,m*n*sizeof(float));
    cudaMalloc(&b_d,m*n*sizeof(float));
    cudaMalloc(&c_d,m*n*sizeof(float));
    
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            a[i*n+j]=rand()%rand_max;
            b[i*n+j]=rand()%rand_max;
        }
    }

    cudaMemcpy(a_d,a,m*n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,m*n*sizeof(float),cudaMemcpyHostToDevice);

    dim3 threads_per_block(16,16);
    dim3 blocks((m+threads_per_block.x-1)/threads_per_block.x,(n+threads_per_block.y-1)/threads_per_block.y);
    mat_add<<<blocks,threads_per_block>>>(a_d,b_d,c_d,m,n);

    cudaMemcpy(c,c_d,m*n*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    for(int i=0;i<m*n;i++)
    {
        std::cout<<a[i]<<" ";
    }
    std::cout<<std::endl;

    for(int i=0;i<m*n;i++)
    {
        std::cout<<b[i]<<" ";
    }
    std::cout<<std::endl;

    for(int i=0;i<m*n;i++)
    {
        std::cout<<c[i]<<" ";
    }
    std::cout<<std::endl;
    free(a);
    free(b);
    free(c);
    return 0;
}