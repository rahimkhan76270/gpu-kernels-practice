#include <cuda_runtime.h>

// tile size
constexpr int T = 32;

__global__ void MatMulTiled(const float* A, const float* B, float* C, int N) {
  __shared__ float sA[T][T], sB[T][T];

  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * T + ty;
  int col = blockIdx.x * T + tx;

  float acc = 0.0f;
  for (int m = 0; m < N; m += T) {
    // load tiles into shared memory
    if (row < N && m+tx < N)  sA[ty][tx] = A[row*N + (m+tx)];
    else                       sA[ty][tx] = 0.0f;
    if (col < N && m+ty < N)  sB[ty][tx] = B[(m+ty)*N + col];
    else                       sB[ty][tx] = 0.0f;
    __syncthreads();

    // compute partial products
    #pragma unroll
    for (int k = 0; k < T; ++k) {
      acc += sA[ty][k] * sB[k][tx];
    }
    __syncthreads();
  }

  if (row < N && col < N) {
    C[row*N + col] = acc;
  }
}

// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n) {   
    dim3 threads(32,32);
    dim3 blocks((n+31)/32,(n+31)/32);
    MatMulTiled<<<blocks,threads>>>(input_a,input_b,output_c,n);
    cudaDeviceSynchronize(); 
}
