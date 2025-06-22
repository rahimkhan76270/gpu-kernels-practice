#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>


__global__ void Conv1DKernel(const float *signal,
                             const float *kernel,
                             float       *out_signal,
                             const int    signal_size,
                             const int    kernel_size,
                             const int    stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < signal_size) {
        out_signal[i] = 0;
        for (int j = -1 * kernel_size / 2; j < kernel_size / 2 + 1; j++) {
            if (i + j >= 0 && i + j < signal_size) {
                out_signal[i] += signal[i + j] * kernel[j + kernel_size / 2];
            }
        }
    }
}

int main()
{
    srand(time(0));
    // int rand_max    = 10;
    int signal_size = 10, out_signal_size, kernel_size = 3, padding = 0, stride = 1, dilation = 1;
    out_signal_size = floor((signal_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1);
    float out_signal[out_signal_size], kernel[kernel_size], *signal_d, *out_signal_d, *kernel_d;
    float signal[signal_size + 2 * padding] = {0};
    // signal
    float sig[] = {0.3661, 0.1468, 0.4150, 0.9767, 0.8751, 0.1556, 0.2482, 0.7860, 0.9778, 0.8273};
    for (int i = 0 + padding; i < signal_size; i++) {
        // signal[i] = rand() % rand_max;
        signal[i] = sig[i];
        std::cout << signal[i] << " ";
    }
    std::cout << std::endl;
    // kernel
    float ker[] = {0.2954, 0.2947, -0.5544};
    for (int i = 0; i < kernel_size; i++) {
        // kernel[i] = rand() % rand_max;
        kernel[i] = ker[i];
        std::cout << kernel[i] << " ";
    }
    std::cout << std::endl;
    // allocate memory
    cudaMalloc(&signal_d, (signal_size + 2 * padding) * sizeof(float));
    cudaMalloc(&out_signal_d, (out_signal_size) * sizeof(float));
    cudaMalloc(&kernel_d, kernel_size * sizeof(float));
    // copy to cuda memory
    cudaMemcpy(signal_d, signal, signal_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_d, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads_per_block(10);
    dim3 num_blocks((threads_per_block.x + signal_size - 1) / threads_per_block.x);

    Conv1DKernel<<<num_blocks, threads_per_block>>>(
        signal_d, kernel_d, out_signal_d, signal_size + 2 * padding, kernel_size, stride);
    cudaDeviceSynchronize();

    cudaMemcpy(out_signal, out_signal_d, out_signal_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(signal_d);
    cudaFree(out_signal_d);
    cudaFree(kernel_d);

    for (int i = 0; i < out_signal_size; i++) {
        std::cout << out_signal[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}