#include "solve.h"
#include <cuda_runtime.h>

__global__ void GausianBlurKernel(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int idx=row*input_cols+col;
    if(row<input_rows && col<input_cols)
    {
        float val=0;
        for(int m=-kernel_rows/2;m<=kernel_rows/2;m++)
        {
            for(int n=-kernel_cols/2;n<=kernel_cols/2;n++)
            {
                bool check=(row+m <input_rows)&&(row+m>=0) && (col+n >=0) && (col+n<input_cols);
                if(check)
                {
                    val+=input[(row+m)*input_cols +col+n]*kernel[(m+kernel_rows/2)*kernel_cols +n+kernel_cols/2];
                }
            }
        }
        output[idx]=val;
    }
}

// input, kernel, output are device pointers
void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    dim3 threads(16,16);
    dim3 blocks((input_cols+threads.x-1)/threads.x,(input_rows+threads.y-1)/threads.y);
    GausianBlurKernel<<<blocks,threads>>>(input,kernel,output,input_rows,input_cols,kernel_rows,kernel_cols);
    cudaDeviceSynchronize();
}
