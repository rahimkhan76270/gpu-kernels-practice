#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr,                                                 \
                    "CUDA error at %s:%d: %s\n",                              \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void MaxReduce(const float* input,
                          float* block_max,
                          int n) {
    extern __shared__ float temp[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    temp[tid] = (idx < n) ? input[idx] : -FLT_MAX;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = temp[tid];
            float b = temp[tid + stride];
            temp[tid] = a > b ? a : b;
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = temp[0];
    }
}

__global__ void expKernel(const float* input,
                          float* output,
                          float input_max,
                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __expf(input[idx] - input_max);
    }
}

__global__ void sumKernel(const float* input,
                          float* output,
                          int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    temp[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, temp[0]);
    }
}

__global__ void softmax_kernel(const float* input,
                               float* output,
                               int n,
                               float sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] / sum;
    }
}

void solve(const float* input, float* output, int N) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* block_max_d = nullptr;
    float* exp_d       = nullptr;
    float* exp_sum_d   = nullptr;

    float* block_max_h = new float[blocksPerGrid];

    CUDA_CHECK(cudaMalloc(&block_max_d, blocksPerGrid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&exp_d,       N             * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&exp_sum_d,   sizeof(float)));

    MaxReduce<<<blocksPerGrid,
                threadsPerBlock,
                threadsPerBlock * sizeof(float)>>>(
        input, block_max_d, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(block_max_h,
                          block_max_d,
                          blocksPerGrid * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(block_max_d));

    float input_max = -FLT_MAX;
    for (int i = 0; i < blocksPerGrid; ++i) {
        if (block_max_h[i] > input_max) {
            input_max = block_max_h[i];
        }
    }

    expKernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, exp_d, input_max, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(exp_sum_d, 0, sizeof(float)));

    sumKernel<<<blocksPerGrid,
                threadsPerBlock,
                threadsPerBlock * sizeof(float)>>>(
        exp_d, exp_sum_d, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float sum_h = 0.0f;
    CUDA_CHECK(cudaMemcpy(&sum_h,
                          exp_sum_d,
                          sizeof(float),
                          cudaMemcpyDeviceToHost));

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        exp_d, output, N, sum_h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(exp_d));
    CUDA_CHECK(cudaFree(exp_sum_d));
    delete[] block_max_h;
}
