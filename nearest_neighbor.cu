#include <cuda_runtime.h>

__global__ void calc_nearest(const float* dist_mat,int *indices,const int n)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<n)
    {
        int min_index=(row==0)?1:0;
        for(int col=1;col<n;col++)
        {
            if(col!=row && dist_mat[row*n+min_index]>dist_mat[row*n+col])
            {
                min_index=col;
            }
        }
        indices[row]=min_index;
    }
}

__global__ void cal_dist_matrix(const float* points,float * dist_mat,const int n)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int idx=row*n+col;
    if(row<n && col<n && row<col)
    {
        float dist=0;
        for(int i=0;i<3;i++)
        {
            dist+=(points[3*row+i]-points[3*col+i])*(points[3*row+i]-points[3*col+i]);
        }
        dist_mat[idx]=dist;
        dist_mat[col*n+row]=dist;
    }
    if(row<n && col<n && row==col) dist_mat[idx]=0;
}
// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    float* dist_mat;
    cudaMalloc(&dist_mat,N*N*sizeof(float));
    dim3 threads(16,16);
    dim3 blocks((N+threads.x-1)/threads.x,(N+threads.y-1)/threads.y);
    cal_dist_matrix<<<blocks,threads>>>(points,dist_mat,N);
    cudaDeviceSynchronize();
    int threads1=256;
    int blocks1=(N+threads1-1)/threads1;
    calc_nearest<<<blocks1,threads1>>>(dist_mat,indices,N);
    cudaDeviceSynchronize();
    cudaFree(dist_mat);
}
