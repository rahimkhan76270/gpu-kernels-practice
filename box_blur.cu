#include <cuda_runtime.h>

__global__ void BlurKernel(const float* input_image, int kernel_size, float* output_image, size_t height, size_t width)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<height && col<width)
    {
        float val=0.0f;
        int val_pixels=0;
        for(int u=-1*kernel_size/2;u<kernel_size/2 +1;u++)
        {
            for(int v=-1*kernel_size/2;v<kernel_size/2 +1;v++)
            {
                if(row+u<height && col+v<width)
                {
                    val+=input_image[(row+u)*width +col+v];
                    val_pixels++;
                }
            }

        }
        output_image[row*width+col]=val/(val_pixels+1e-8);
    }
}

// Note: input_image, output_image are all device pointers to float32 arrays
extern "C" void solution(const float* input_image, int kernel_size, float* output_image, size_t height, size_t width) {    
    dim3 threads(16,16);
    dim3 blocks((width+threads.x-1)/threads.x,(height+threads.y-1)/threads.y);
    BlurKernel<<<blocks,threads>>>(input_image,kernel_size,output_image,height,width);
    cudaDeviceSynchronize();
}
