// #include <cuda_runtime.h>

// constexpr int T=8;

// __global__ void batch_mat_mul_tiled(const float* A,const float *B,float *C,int BATCH,int m,int n,int k)
// {
//     int bx=blockIdx.x;
//     int by=blockIdx.y;
//     int bz=blockIdx.z;
//     int tx=threadIdx.x;
//     int ty=threadIdx.y;
//     int tz=threadIdx.z;

//     int batch=bx*T+tx;
//     int row=by*T+ty;
//     int col=bz*T+tz;
//     int num_tiles=ceil(static_cast<float>(k)/T);
//     __shared__ float sh_a[T][T];
//     __shared__ float sh_b[T][T];
//     float val=0.0f;
//     for(int tile=0;tile<num_tiles;tile++)
//     {
//         if(row<m && tile*T+tz<k)
//         {
//             sh_a[ty][tz]=A[batch*m*k+row*k+tile*T+tz];
//         }
//         else
//         {
//             sh_a[ty][tz]=0.0f;
//         }
//         if(col<n && tile*T+ty<k)
//         {
//             sh_b[ty][tz]=B[batch*k*n+(tile*T+ty)*n+col];
//         }
//         else
//         {
//             sh_b[ty][tz]=0.0f;
//         }
//         __syncthreads();
//         for(int k_tile=0;k_tile<T;k_tile++)
//         {
//             val+=sh_a[ty][k_tile]*sh_b[k_tile][tz];
//         }
//         __syncthreads();
//     }
//     if(batch<BATCH && row<m && col<n)
//     {
//         C[batch*m*n+row*n+col]=val;
//     }
// }

// __global__ void batch_matrix_multilpication(const float* A,const float *B,float *C,int BATCH,int m,int n,int k)
// {
//     int batch=blockIdx.x*blockDim.x+threadIdx.x;
//     int row=blockIdx.y*blockDim.y+threadIdx.y;
//     int col=blockIdx.z*blockDim.z+threadIdx.z;
//     int idx=batch*m*n+row*n+col;
//     if(batch<BATCH && row<m && col<n){
//         float val=0.0f;
//         for(int x=0;x<k;x++)
//         {
//             val+=A[batch*m*k+row*k+x]*B[batch*k*n+x*n+col];
//         }
//         C[idx]=val;
//     }
    
// }

// // A, B, C are device pointers
// extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
//     dim3 threads(8,8,8);
//     dim3 blocks((BATCH+threads.x-1)/threads.x,(M+threads.y-1)/threads.y,(N+threads.z-1)/threads.z);
//     // batch_matrix_multilpication<<<blocks,threads>>>(A,B,C,BATCH,M,N,K);
//     batch_mat_mul_tiled<<<blocks,threads>>>(A,B,C,BATCH,M,N,K);
//     cudaDeviceSynchronize();
// } 
#include <cuda_runtime.h>

constexpr int T = 8;

__global__ void batch_mat_mul_tiled(
    const float * __restrict__ A,
    const float * __restrict__ B,
          float *       C,
    int BATCH, int M, int N, int K)
{
    // batch = blockIdx.x
    int batch = blockIdx.x;

    // within‐tile row/col
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // global row/col of the C‐matrix
    int row = blockIdx.y * T + ty;
    int col = blockIdx.z * T + tx;

    // shared tiles
    __shared__ float sA[T][T];
    __shared__ float sB[T][T];

    float acc = 0.0f;

    // integer ceil(K/T)
    int nTiles = (K + T - 1) / T;

    for (int t = 0; t < nTiles; ++t) {
        // load A(tileRow = row, tileCol = t*T + tx)
        int aCol = t * T + tx;
        if (batch < BATCH && row < M && aCol < K) {
            sA[ty][tx] = A[ batch * (M * K)
                          + row  *    K
                          + aCol        ];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // load B(tileRow = t*T + ty, tileCol = col)
        int bRow = t * T + ty;
        if (batch < BATCH && bRow < K && col < N) {
            sB[ty][tx] = B[ batch * (K * N)
                          + bRow  *    N
                          + col        ];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // accumulate the inner‐product for this tile
        #pragma unroll
        for (int k0 = 0; k0 < T; ++k0) {
            acc += sA[ty][k0] * sB[k0][tx];
        }

        __syncthreads();
    }

    // write‐back
    if (batch < BATCH && row < M && col < N) {
        C[ batch * (M * N)
         + row   *    N
         + col      ] = acc;
    }
}

// host wrapper
extern "C"
void solve(
    const float* A,
    const float* B,
          float* C,
    int BATCH, int M, int N, int K)
{
    // block: T×T threads
    dim3 blockDim(T, T, 1);

    // grid: one block per (batch, row-tile, col-tile)
    dim3 gridDim(
      BATCH,
      (M + T - 1) / T,
      (N + T - 1) / T
    );

    batch_mat_mul_tiled<<<gridDim, blockDim>>>(A, B, C, BATCH, M, N, K);
    cudaDeviceSynchronize();
}
