#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

__global__ void RowWiseMean(const float *arr, float *mean, int R, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < R) {
        mean[i] = 0;
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < C; j++) {
            mean[i] += arr[i * C + j] / C;
        }
    }
}

__global__ void RowWiseVariance(const float *arr, const float *mean, float *var, int R, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < R) {
        var[i] = 0;
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < C; j++) {
            float diff = arr[i * C + j] - mean[i];
            var[i] += (diff * diff) / C;
        }
    }
}

__global__ void LayerNormKernel(const float *arr,
                                const float *mean,
                                const float *var,
                                float       *norm,
                                int          R,
                                int          C,
                                float        scale,
                                float        bias,
                                float        eps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < R && j < C) {
        norm[i * C + j] = (arr[i * C + j] - mean[i]) / sqrt(var[i] + eps);
    }
}

int main()
{
    cudaEvent_t start, stop;
    float       milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    srand(time(0));
    int   rand_max = 10;
    int   R = 500, C = 500;
    float arr[R * C], l_norm[R * C], *arr_d, *l_norm_d, *mean_d, *var_d;

    std::cout << "[\n";
    for (int i = 0; i < R; i++) {
        // std::cout << "[ ";
        for (int j = 0; j < C; j++) {
            arr[i * C + j] = rand() % rand_max;
            // std::cout << arr[i * R + j] << ", ";
        }
        // std::cout << "]," << std::endl;
    }
    std::cout << "]\n";
    cudaEventRecord(start);

    cudaMalloc(&arr_d, R * C * sizeof(float));
    cudaMalloc(&l_norm_d, R * C * sizeof(float));
    cudaMalloc(&mean_d, R * sizeof(float));
    cudaMalloc(&var_d, R * sizeof(float));

    cudaMemcpy(arr_d, arr, R * C * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads_per_block(16, 16);
    dim3 num_blocks((R + threads_per_block.x - 1) / threads_per_block.x,
                    (C + threads_per_block.y - 1) / threads_per_block.y);

    RowWiseMean<<<num_blocks, threads_per_block>>>(arr_d, mean_d, R, C);
    cudaDeviceSynchronize();
    RowWiseVariance<<<num_blocks, threads_per_block>>>(arr_d, mean_d, var_d, R, C);
    cudaDeviceSynchronize();
    LayerNormKernel<<<num_blocks, threads_per_block>>>(arr_d, mean_d, var_d, l_norm_d, R, C, 1, 0, 1e-5);
    cudaDeviceSynchronize();
    cudaMemcpy(l_norm, l_norm_d, R * C * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(arr_d);
    cudaFree(l_norm_d);
    cudaFree(mean_d);
    cudaFree(var_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // for (int i = 0; i < R; i++) {
    //     for (int j = 0; j < C; j++) {
    //         std::cout << l_norm[i * R + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken by kernels: " << milliseconds << " ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}