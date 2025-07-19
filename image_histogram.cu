#include <cuda_runtime.h>

__global__ void ImageHistKernel(const float* image, int num_bins, float* histogram, size_t height, size_t width)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<height && col<width)
    {
        int val=ceil(image[row*width+col]);
        if (val >= 0 && val < num_bins) {
            atomicAdd(&histogram[val], 1.0f);
        }
    }
}

// Note: image, histogram are all device pointers to float32 arrays
extern "C" void solution(const float* image, int num_bins, float* histogram, size_t height, size_t width) {    
    dim3 threads(16,16);
    dim3 blocks((width+threads.x-1)/threads.x,(height+threads.y-1)/threads.y);
    ImageHistKernel<<<blocks,threads>>>(image,num_bins,histogram,height,width);
    cudaDeviceSynchronize();
}
