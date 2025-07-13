#include <cuda_runtime.h>
#define TILE_WIDTH 16

__global__ void ImageThreshKernel(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int idx=row*width+col;
    if(row<height && col<width)
    {
        output_image[idx]=(input_image[idx]>threshold_value)?255.0f:0.0f;
    }
}
// Note: input_image, output_image are all device pointers to float32 arrays
extern "C" void solution(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width) {    
    dim3 threads(TILE_WIDTH,TILE_WIDTH);
    dim3 blocks((width+TILE_WIDTH-1)/TILE_WIDTH,(height+TILE_WIDTH-1)/TILE_WIDTH);
    ImageThreshKernel<<<blocks,threads>>>(input_image,threshold_value,output_image,height,width);
    cudaDeviceSynchronize();
}
