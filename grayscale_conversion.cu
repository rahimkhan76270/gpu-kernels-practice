#include <cuda_runtime.h>

__global__ void GrayScaleKernel(const float* rgb_image, float* grayscale_output, size_t height, size_t width, size_t channels)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int idx=row*width+col;
    if(row<height && col<width && idx+2<height*width*channels)
    {
        float R=rgb_image[channels*idx];
        float G=rgb_image[channels*idx+1];
        float B=rgb_image[channels*idx+2];
        grayscale_output[idx]=0.299*R+0.587*G+0.114*B;
    }
}
// Note: rgb_image, grayscale_output are all device pointers to float32 arrays
extern "C" void solution(const float* rgb_image, float* grayscale_output, size_t height, size_t width, size_t channels) {    
    dim3 threads(16,16);
    dim3 blocks((width+threads.x-1)/threads.x,(height+threads.y-1)/threads.y);
    GrayScaleKernel<<<blocks,threads>>>(rgb_image,grayscale_output,height,width,channels);
    cudaDeviceSynchronize();
}
