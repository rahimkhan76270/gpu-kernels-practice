#include <cuda_runtime.h>

__global__ void SumOverDimKernel(const float* input, int dim, float* output, size_t ndim,size_t* strides,size_t*strides_out,size_t total_elems)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<total_elems)
    {
        size_t float_out_idx=0;
        size_t temp=idx;
        for(int i=0;i<ndim;i++)
        {
            size_t coord=temp/strides[i];
            temp%=strides[i];
            if(i!=dim) float_out_idx+=coord*strides_out[i];
        }
        atomicAdd(&output[float_out_idx],input[idx]);
    }
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
    int threads=256;
    int blocks=(total_elems+threads-1)/threads;
    SumOverDimKernel<<<blocks,threads>>>(input,dim,output,ndim,strides_d,strides_out_d,total_elems);
    cudaDeviceSynchronize();
}
