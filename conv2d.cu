#include "solve.h"
#include <cuda_runtime.h>

__global__ void Conv2DKernel(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols,
           int output_rows,int output_cols)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int idx=row*output_cols+col;
    if(row<output_rows && col<output_cols)
    {
        float val=0;
        for(int m=0;m<kernel_rows;m++)
        {
            for(int n=0;n<kernel_cols;n++)
            {
                bool check=(row+m <input_rows) && (col+n<input_cols);
                if(check)
                {
                    val+=input[(row+m)*input_cols +col+n]*kernel[m*kernel_cols +n];
                }
            }
        }
        output[idx]=val;
    }
}
// input, kernel, output are device pointers
void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    dim3 threads(16,16);
    dim3 blocks((output_cols+threads.x-1)/threads.x,(output_rows+threads.y-1)/threads.y);
    Conv2DKernel<<<blocks,threads>>>(input,kernel,output,input_rows,input_cols,kernel_rows,kernel_cols,output_rows,output_cols);
    cudaDeviceSynchronize();
}
