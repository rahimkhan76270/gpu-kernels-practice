#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>
#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err = call;                                                            \
        if (err != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s",                                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));                          \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

__global__ void SumOverDimKernel(const float* input, int dim, float* output, size_t* shape, size_t ndim, size_t* strides, size_t total_elems,size_t* strides_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    size_t temp = idx;
    size_t flat_out_idx = 0;
    for (size_t i = 0; i < ndim; i++) {
        size_t coord = temp / strides[i];
        temp = temp % strides[i];
        if (i != (size_t)dim) flat_out_idx += coord * strides_out[i];
    }
    atomicAdd(&output[flat_out_idx], input[idx]);
}

// Note: input, output, shape are all device pointers to float32 arrays
extern "C" void solution(const float* input, int dim, float* output, size_t* shape, size_t ndim) {   
    size_t* shape_h = new size_t[ndim];
    cudaMemcpy(shape_h, shape, ndim * sizeof(size_t), cudaMemcpyDeviceToHost);
    size_t total_elems = 1;
    for (int i = 0; i < ndim; i++) {
        total_elems *= shape_h[i];
    } 
    size_t *strides_out=new size_t[ndim];
    size_t *shape_out=new size_t[ndim];
    std::copy(shape_h,shape_h+ndim,shape_out);
    shape_out[dim]=1;
    strides_out[ndim-1]=1;
    size_t* strides = new size_t[ndim];
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape_h[i + 1];
        strides_out[i]=strides_out[i+1]*shape_out[i+1];
    }
    
    size_t* strides_d,*strides_out_d;
    CUDA_CHECK(cudaMalloc(&strides_d, ndim * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&strides_out_d, ndim * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(strides_d, strides, ndim * sizeof(size_t), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(strides_out_d, strides_out, ndim * sizeof(size_t), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemset(output, 0, (total_elems / shape_h[dim]) * sizeof(float)));
    int threads = 512;
    int blocks = (total_elems + threads - 1) / threads;
    SumOverDimKernel<<<blocks, threads>>>(input, dim, output, shape, ndim, strides_d, total_elems,strides_out_d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(strides_d));
    CUDA_CHECK(cudaFree(strides_out_d));
    delete[] strides;
    delete[] shape_h;
}
