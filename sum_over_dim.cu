#include <cuda_runtime.h>

// __global__ void SumOverDimKernel(const float* input, int dim, float* output, size_t ndim,size_t* strides,size_t*strides_out,size_t total_elems)
// {
//     int idx=blockIdx.x*blockDim.x+threadIdx.x;
//     if(idx<total_elems)
//     {
//         size_t float_out_idx=0;
//         size_t temp=idx;
//         for(int i=0;i<ndim;i++)
//         {
//             size_t coord=temp/strides[i];
//             temp%=strides[i];
//             if(i!=dim) float_out_idx+=coord*strides_out[i];
//         }
//         atomicAdd(&output[float_out_idx],input[idx]);
//     }
// }
__global__ void sum_over_dim_gather(
    const float* __restrict__ input,
    float*              output,
    int                 dim,
    size_t              shape_dim,      // shape[dim]
    size_t              total_out_elems,
    size_t              ndim,
    const size_t* __restrict__ strides_in,
    const size_t* __restrict__ strides_out)
{
    size_t out_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elems) return;

    // 1) Reconstruct input base index from out_idx (skipping 'dim')
    size_t temp = out_idx;
    size_t base = 0;
    for (int d = 0; d < ndim; ++d) {
        if (d == dim) continue;
        size_t coord = temp / strides_out[d];
        temp       %= strides_out[d];
        base       += coord * strides_in[d];
    }

    // 2) Sum across the reduced dimension
    float s = 0.0f;
    size_t stride = strides_in[dim];
    #pragma unroll
    for (int j = 0; j < shape_dim; ++j) {
        s += input[base + j * stride];
    }

    // 3) Single write
    output[out_idx] = s;
}

// Note: input, output, shape are all device pointers to float32 arrays
extern "C" void solution(const float* input, int dim, float* output, size_t* shape, size_t ndim) {    
    size_t *shape_h=new size_t[ndim];
    size_t *shape_out_h=new size_t[ndim];
    cudaMemcpy(shape_h,shape,ndim*sizeof(size_t),cudaMemcpyDeviceToHost);
    cudaMemcpy(shape_out_h,shape,ndim*sizeof(size_t),cudaMemcpyDeviceToHost);
    shape_out_h[dim]=1;
    size_t* strides=new size_t[ndim];
    size_t* strides_out=new size_t[ndim];
    strides[ndim-1]=1;
    strides_out[ndim-1]=1;
    for(int i=ndim-2;i>=0;i--)
    {
        strides[i]=strides[i+1]*shape_h[i+1];
        strides_out[i]=strides_out[i+1]*shape_out_h[i+1];
    }
    size_t total_elems=1;
    for(int i=0;i<ndim;i++)
    {
        total_elems*=shape_h[i];
    }
    size_t total_elems_out=total_elems/shape_h[dim];
    size_t *strides_d,*strides_out_d;
    cudaMalloc(&strides_d,ndim*sizeof(size_t));
    cudaMalloc(&strides_out_d,ndim*sizeof(size_t));
    cudaMemcpy(strides_d,strides,ndim*sizeof(size_t),cudaMemcpyHostToDevice);
    cudaMemcpy(strides_out_d,strides_out,ndim*sizeof(size_t),cudaMemcpyHostToDevice);
    cudaMemset(output,1,total_elems_out*sizeof(float));
    int    threads         = 256;
    int    blocks          = (total_elems_out + threads - 1) / threads;

    sum_over_dim_gather
    <<<blocks, threads>>>(
        input, output, dim,
        shape_h[dim], total_elems_out,
        ndim, strides_d, strides_out_d);
        cudaDeviceSynchronize();
}
