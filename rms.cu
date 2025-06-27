#include <cuda_runtime.h>

__global__ void compute_squared_errors(
    const float* preds, const float* targets, float* errors, int N) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float diff = preds[idx] - targets[idx];
        errors[idx] = diff * diff;
    }
}

__global__ void reduce_sum(const float* input, float* output, int N) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    temp[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            temp[tid] += temp[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = temp[0];
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t* shape, size_t ndim) {
    // 1. Compute total number of elements
    int total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= shape[i];
    }

    // 2. Launch kernel to compute squared errors
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    float* errors_d;
    cudaMalloc(&errors_d, total_elements * sizeof(float));
    compute_squared_errors<<<blocks, threads>>>(predictions, targets, errors_d, total_elements);

    // 3. Launch block-level reductions
    float* partial_sums_d;
    cudaMalloc(&partial_sums_d, blocks * sizeof(float));
    reduce_sum<<<blocks, threads>>>(errors_d, partial_sums_d, total_elements);

    // 4. Copy partial sums back to host and compute final MSE
    float* partial_sums_h = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(partial_sums_h, partial_sums_d, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        total += partial_sums_h[i];
    }

    float mse = total / total_elements;

    // 5. Copy result to device output
    cudaMemcpy(output, &mse, sizeof(float), cudaMemcpyHostToDevice);

    // Cleanup
    cudaFree(errors_d);
    cudaFree(partial_sums_d);
    free(partial_sums_h);
}
