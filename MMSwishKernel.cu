#include <cuda_runtime.h>
#define TILE_WIDTH 16

__device__ float sigmoid(const float x)
{
    float val=1/(1+exp(-x));
    return val;
}


__global__ void MMSwishActivation(const float* input_matrix, const float* weight_matrix, const float* bias, float scaling_factor, float* output, size_t batch_size, size_t in_features, size_t out_features)
{
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int num_tiles=ceil(static_cast<float>(in_features)/TILE_WIDTH);

    int row=by*TILE_WIDTH+ty;
    int col=bx*TILE_WIDTH+tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];
    float val=0;
    for(int tile=0;tile<num_tiles;tile++)
    {
        //load input
        if(row<batch_size && tile*TILE_WIDTH+tx<in_features)
        {
            sh_A[ty][tx]=input_matrix[row*in_features+tile*TILE_WIDTH+tx];
        }
        else
        {
            sh_A[ty][tx]=0.0f;
        }
        // load weight
        if(col<out_features && tile*TILE_WIDTH+ty<in_features)
        {
            sh_B[ty][tx]=weight_matrix[col*in_features +tile*TILE_WIDTH+ty];
        }
        else{
            sh_B[ty][tx]=0.0f;
        }
        __syncthreads();
        for(int k_tile=0;k_tile<TILE_WIDTH;k_tile++)
        {
            val+=sh_A[ty][k_tile]*sh_B[k_tile][tx];
        }
        __syncthreads();
    }
    if(row<batch_size && col<out_features)
    {
        output[row*out_features+col]=scaling_factor*(val +bias[col])*sigmoid(val +bias[col]);
    }
}

// Note: input_matrix, weight_matrix, bias, output are all device pointers to float32 arrays
extern "C" void solution(const float* input_matrix, const float* weight_matrix, const float* bias, float scaling_factor, float* output, size_t batch_size, size_t in_features, size_t out_features) {    
    dim3 threads(TILE_WIDTH,TILE_WIDTH);
    dim3 blocks((out_features+TILE_WIDTH-1)/TILE_WIDTH,(batch_size+TILE_WIDTH-1)/TILE_WIDTH);
    MMSwishActivation<<<blocks,threads>>>(input_matrix,weight_matrix,bias,scaling_factor,output,batch_size,in_features,out_features);
    cudaDeviceSynchronize();
}
