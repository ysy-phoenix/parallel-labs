#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>

const int LENGTH = 2048;
const int SIZE = LENGTH * LENGTH;

// 单block多threads
__global__ void matrix_mul1(float *out, float *a, float *b, int n) 
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    for (int i = tid_y; i < n; i += blockDim.y) {
        for (int j = tid_x; j < n; j += blockDim.x) {
            float sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

// 多blocks多threads
__global__ void matrix_mul2(float *out, float *a, float *b, int n) 
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    int x = bid_x * blockDim.x + tid_x;
    int y = bid_y * blockDim.y + tid_y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int i = y; i < n; i += stride_y) {
        for (int j = x; j < n; j += stride_x) {
            float sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

// 共享内存
__global__ void matrix_mul3(float *out, float *a, float *b, int n) 
{
    __shared__ float temp_a[32][32];
    __shared__ float temp_b[32][32];

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    int x = bid_x * blockDim.x + tid_x;
    int y = bid_y * blockDim.y + tid_y;

    float temp_f = 0.0;
    for (int block = 0; block < n / 32; block++) {
        int offset_a = bid_y * blockDim.x * n + block * blockDim.x;
        int offset_b = block * blockDim.x * n + bid_x * blockDim.x;
        temp_a[tid_y][tid_x] = a[offset_a + tid_y * n + tid_x];
        temp_b[tid_y][tid_x] = b[offset_b + tid_y * n + tid_x];

        __syncthreads();

        for (int k = 0; k < 32; k++) {
            temp_f += temp_a[tid_y][k] * temp_b[k][tid_x];
        }
        
        __syncthreads();
    }

    out[y * n + x] = temp_f;
}

int correctness(float *serial, float *paralell) {
    for (int i = 0; i < SIZE; i++) {
        if (serial[i] != paralell[i]) {
            printf("%f, %f\n", serial[i], paralell[i]);
            return 0;
        }  
    }
    return 1;
}

int main()
{
    float *a, *b, *out, *out1, *out2, *out3;
    float *d_a, *d_b, *d_out1, *d_out2, *d_out3; 
    printf("Matrix Length is: %dx%d\n", LENGTH, LENGTH);

    //===================步骤1===================
    // Allocate memory on CPU
    a = (float*)malloc(sizeof(float) * SIZE);
    b = (float*)malloc(sizeof(float) * SIZE);
    out = (float*)malloc(sizeof(float) * SIZE);
    out1 = (float*)malloc(sizeof(float) * SIZE);
    out2 = (float*)malloc(sizeof(float) * SIZE);
    out3 = (float*)malloc(sizeof(float) * SIZE);
 
    // data initializtion
    for (int i = 0; i < SIZE; i++) {
        a[i] = i % (LENGTH - 2) - 5.0;
        b[i] = i % (LENGTH + 2) + 5.0;
    }
    clock_t start0 = clock();
    for (int i = 0; i < LENGTH; i++) {
        for (int j = 0; j < LENGTH; j++) {
            float sum = 0.0;
            for (int k = 0; k < LENGTH; k++) {
                sum += a[i * LENGTH + k] * b[k * LENGTH + j];
            }
            out[i * LENGTH + j] = sum;
        }
    }
    clock_t finish0 = clock();
    printf("serial running: %f milliseconds\n", (double)(finish0 - start0) / 1000);

    //===================步骤1===================
    // Allocate memory on GPU
    cudaMalloc((void**)&d_a, sizeof(float) * SIZE);
    cudaMalloc((void**)&d_b, sizeof(float) * SIZE);
    cudaMalloc((void**)&d_out1, sizeof(float) * SIZE);
    cudaMalloc((void**)&d_out2, sizeof(float) * SIZE);
    cudaMalloc((void**)&d_out3, sizeof(float) * SIZE);
 
    //===================步骤2===================
    // copy operator to GPU
    cudaMemcpy(d_a, a, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
    
    //===================步骤3===================
    // GPU do the work, CPU waits
    cudaEvent_t start, stop;
    float time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 单block多threads
    cudaEventRecord(start, 0);
    dim3 blockSize(16, 16);
    matrix_mul1<<<1,blockSize>>>(d_out1, d_a, d_b, LENGTH);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("parallel running(one block multiple threads): %f milliseconds\n", time);
    
    // 多blocks多threads
    cudaEventRecord(start, 0);
    dim3 gridSize(16, 16);
    matrix_mul2<<<gridSize,blockSize>>>(d_out2, d_a, d_b, LENGTH);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("parallel running(multiple blocks multiple threads): %f milliseconds\n", time);

    // 共享内存
    cudaEventRecord(start, 0);
    blockSize = dim3(32, 32);
    gridSize = dim3(LENGTH / 32, LENGTH / 32);
    matrix_mul3<<<gridSize,blockSize>>>(d_out3, d_a, d_b, LENGTH);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("parallel running(shared memory): %f milliseconds\n", time);

    //===================步骤4===================
    // Get results from the GPU
    cudaMemcpy(out1, d_out1, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(out2, d_out2, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(out3, d_out3, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);
    printf("correctness(one block multiple threads): %d\n", correctness(out, out1));
    printf("correctness(multiple blocks multiple threads): %d\n", correctness(out, out2));
    printf("correctness(shared memory): %d\n", correctness(out, out3));
    
    //===================步骤5===================
    // Free the memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_out3);

    free(a);
    free(b);
    free(out);
    free(out1);
    free(out2);
    free(out3);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}