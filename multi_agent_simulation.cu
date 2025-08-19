#include <cuda_runtime.h>

__global__ void FindNextPosition(const float *agents,const float* neighbors,float * agents_next,const int N)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N)
    {
        // Calculate v_avg
        float v_avg_x=0.0f;
        float v_avg_y=0.0f;
        int num_neighbors=0;
        for(int i=0;i<N;i++)
        {
            if(neighbors[idx*N+i] > 0.5f)
            {
                v_avg_x+=agents[4*i+2];
                v_avg_y+=agents[4*i+3];
                num_neighbors++;
            }
        }
        if(num_neighbors>0)
        {
            v_avg_x/=num_neighbors;
            v_avg_y/=num_neighbors;
        }
        else {
            v_avg_x = agents[4*idx+2];
            v_avg_y = agents[4*idx+3];
        }

        float v_new_x=0.05f*(v_avg_x - agents[4*idx+2]) + agents[4*idx+2];
        float v_new_y=0.05f*(v_avg_y - agents[4*idx+3]) + agents[4*idx+3];

        float new_pos_x=agents[4*idx]   + v_new_x;
        float new_pos_y=agents[4*idx+1] + v_new_y;

        agents_next[4*idx]   = new_pos_x;
        agents_next[4*idx+1] = new_pos_y;
        agents_next[4*idx+2] = v_new_x;
        agents_next[4*idx+3] = v_new_y;
    }
}

__global__ void IdentifyNeighbors(const float *agents,float * neighbors,const int N)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N)
    {
        float x=agents[4*idx];
        float y=agents[4*idx+1];
        for(int i=idx+1;i<N;i++)
        {
            float dx = x - agents[4*i];
            float dy = y - agents[4*i+1];
            float dist = dx*dx + dy*dy;
            if(dist < 25.0f)  // r^2
            {
                neighbors[idx*N+i] = 1.0f;
                neighbors[i*N+idx] = 1.0f;
            }
        } 
    }
}

// agents, agents_next are device pointers
extern "C" void solve(const float* agents, float* agents_next, int N) {
    int threads=256;
    int blocks=(N+threads-1)/threads;
    float *neighbors_d;
    cudaMalloc(&neighbors_d,N*N*sizeof(float));
    cudaMemset(neighbors_d,0,N*N*sizeof(float));   

    IdentifyNeighbors<<<blocks,threads>>>(agents,neighbors_d,N);
    cudaDeviceSynchronize();
    FindNextPosition<<<blocks,threads>>>(agents,neighbors_d,agents_next,N);
    cudaDeviceSynchronize();

    cudaFree(neighbors_d);
}
