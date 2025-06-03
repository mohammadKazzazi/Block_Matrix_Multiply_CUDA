#include "bmm.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// Keep your original tile sizes that showed better performance
#define TILEX 32
#define TILEY 8
#define TILEZ 32

__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {
    // Shared memory
    __shared__ float AS[TILEY][TILEZ];
    __shared__ float BS[TILEZ][TILEX];
    
    // Calculate global indices
    const int i = TILEY * by + ty;
    const int j = TILEX * bx + tx;
    
    float temp_sum = 0.0f;
    
    // Precompute loop bound
    const int num_tiles = n / TILEZ;
    
    for (int p = 0; p < num_tiles; p++) {
        // Optimized loading of AS
        if (TILEX > TILEZ) {
            if (tx < TILEZ) {
                AS[ty][tx] = ad[i * n + (p * TILEZ + tx)];
            }
        } else {
            #pragma unroll
            for (int k = 0; k < TILEZ/TILEX; k++) {
                AS[ty][k * TILEX + tx] = ad[i * n + (p * TILEZ + k * TILEX + tx)];
            }
        }
        
        // Optimized loading of BS
        if (TILEZ > TILEY) {
            #pragma unroll
            for (int k = 0; k < TILEZ/TILEY; k++) {
                BS[k * TILEY + ty][tx] = bd[(p * TILEZ + k * TILEY + ty) * n + j];
            }
        } else {
            if (ty < TILEZ) {
                BS[ty][tx] = bd[(p * TILEZ + ty) * n + j];
            }
        }
        
        __syncthreads();
        
        // Accumulate product with loop unrolling
        #pragma unroll
        for (int q = 0; q < TILEZ; q++) {
            temp_sum += AS[ty][q] * BS[q][tx];
        }
        
        __syncthreads();
    }
    
    // Only write if within bounds
    if (i < n && j < n) {
        cd[i * n + j] = temp_sum;
    }
}

dim3 getDimGrid(const int m, const int n) {
    dim3 dimGrid((n + TILEX - 1) / TILEX, (n + TILEY - 1) / TILEY);
    return dimGrid;
}

dim3 getDimBlock(const int m, const int n) {
    dim3 dimBlock(TILEX, TILEY);
    return dimBlock;
}