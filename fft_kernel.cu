#include <cuda_runtime.h>
#include <cmath>
#define PI 3.141592653589793f

__global__ void FFTKernel(const float* signal, float* spectrum, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        float real = 0.0f;
        float imag = 0.0f;
        for(int i = 0; i < N; i++)
        {
            float real_part = signal[2*i];
            float imag_part = signal[2*i+1];
            float angle = -2.0f * PI * idx * i / N;
            float c = cosf(angle);
            float s = sinf(angle);
            real += real_part * c - imag_part * s;
            imag += real_part * s + imag_part * c;
        }
        spectrum[2*idx]   = real;
        spectrum[2*idx+1] = imag;
    }
}

// signal and spectrum are device pointers
extern "C" void solve(const float* signal, float* spectrum, int N) {
    int threads=256;
    int blocks=(N +threads-1)/threads;
    FFTKernel<<<blocks,threads>>>(signal,spectrum,N);
    cudaDeviceSynchronize();
}
