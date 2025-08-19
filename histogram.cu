#include "solve.h"
#include <cuda_runtime.h>
#define COARSE_FACTOR 8

__global__ void ImageHistKernel(const int* input, int* histogram, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < COARSE_FACTOR; ++i) {
        int idx = tid * COARSE_FACTOR + i;
        if (idx < N) {
            atomicAdd(&histogram[input[idx]], 1);
        }
    }
}
// input, histogram are device pointers
void solve(const int* input, int* histogram, int N, int num_bins) {
    cudaMemset(histogram, 0, sizeof(int) * num_bins);
    int threads = 256;
    int total_work_units = (N + COARSE_FACTOR - 1) / COARSE_FACTOR;
    int blocks = (total_work_units + threads - 1) / threads;
    ImageHistKernel<<<blocks, threads>>>(input, histogram, N);
    cudaDeviceSynchronize();
}
