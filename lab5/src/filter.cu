#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>   

using namespace cv;

void printDeviceProp(const cudaDeviceProp prop) {
    printf("Device Name: %s\n", prop.name);
    printf("totalGlobalMem: %.0f MBytes---%ld Bytes\n", (float)prop.totalGlobalMem/1024/1024, prop.totalGlobalMem);
    printf("sharedMemPerBlock: %lu\n", prop.sharedMemPerBlock);
    printf("regsPerBolck: %d\n", prop.regsPerBlock);
    printf("warpSize: %d\n", prop.warpSize);
    printf("memPitch: %lu\n", prop.memPitch);
    printf("maxTreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0-2]: %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0-2]: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem: %lu\n", prop.totalConstMem);
    printf("major.minor: %d.%d\n", prop.major, prop.minor);
    printf("clockRate: %d\n", prop.clockRate);
    printf("textureAlignment: %lu\n", prop.textureAlignment);
    printf("deviceOverlap: %d\n", prop.deviceOverlap);
    printf("multiProcessorCount: %d\n", prop.multiProcessorCount);
    printf("===========================\n");
}

void serial_mean_filter(unsigned char *img, unsigned char *filter, int width, int height) {
    for (int i = 0; i < height; i++) {
        if (i <= 0 || i >= height - 1)
            continue;
        for (int j = 0; j < width; j++) {
            if (j <= 0 || j >= width - 1) 
                continue;
            float temp_f = 0.0;
            temp_f += img[(j - 1) * width + i - 1];
            temp_f += img[(j - 1) * width + i];
            temp_f += img[(j - 1) * width + i + 1];
            temp_f += img[j * width + i - 1];
            temp_f += img[j * width + i];
            temp_f += img[j * width + i + 1];
            temp_f += img[(j + 1) * width + i - 1];
            temp_f += img[(j + 1) * width + i];
            temp_f += img[(j + 1) * width + i + 1];
            temp_f /= 9;
            filter[j * width + i] = (unsigned char) temp_f; 
        }
    }
}

// 单block多threads
__global__ void mean_filter1(unsigned char *img, unsigned char *filter, int width, int height) {
    float temp_f;
    
    for(int y = threadIdx.y; y < height; y += blockDim.y) {
        if (y <= 0 || y >= height - 1)
            continue;
        for (int x = threadIdx.x; x < width; x += blockDim.x) {
            if (x <= 0 || x >= width - 1) 
                continue;
            temp_f = 0.0;
            temp_f += img[(y - 1) * width + x - 1];
            temp_f += img[(y - 1) * width + x];
            temp_f += img[(y - 1) * width + x + 1];
            temp_f += img[y * width + x - 1];
            temp_f += img[y * width + x];
            temp_f += img[y * width + x + 1];
            temp_f += img[(y + 1) * width + x - 1];
            temp_f += img[(y + 1) * width + x];
            temp_f += img[(y + 1) * width + x + 1];
            temp_f /= 9;
            filter[y * width + x] = (unsigned char) temp_f;
        }
    }
}

// 多blocks多threads
__global__ void mean_filter2(unsigned char *img, unsigned char *filter, int width, int height) {
    float temp_f;
    
    for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += gridDim.y * blockDim.y) {
        if (y <= 0 || y >= height - 1)
            continue;
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += gridDim.x * blockDim.x) {
            if (x <= 0 || x >= width - 1) 
                continue;
            temp_f = 0.0;
            temp_f += img[(y - 1) * width + x - 1];
            temp_f += img[(y - 1) * width + x];
            temp_f += img[(y - 1) * width + x + 1];
            temp_f += img[y * width + x - 1];
            temp_f += img[y * width + x];
            temp_f += img[y * width + x + 1];
            temp_f += img[(y + 1) * width + x - 1];
            temp_f += img[(y + 1) * width + x];
            temp_f += img[(y + 1) * width + x + 1];
            temp_f /= 9;
            filter[y * width + x] = (unsigned char) temp_f;
        }
    }
}

// 使用共享内存
__global__ void mean_filter3(unsigned char *img, unsigned char *filter, int width, int height) {
    __shared__ unsigned char img_shared[18][18];
    float temp_f;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y > 0 && y < height && x > 0 && x < width) {
        img_shared[tid_y + 1][tid_x + 1] = img[y * width + x];
    }

    if (tid_x == 0 && x > 0) {
        img_shared[tid_y + 1][0] = img[y * width + x - 1];
    }

    if (tid_x == 15 && x < width - 1) {
        img_shared[tid_y + 1][17] = img[y * width + x + 1];
    }

    if (tid_y == 0 && y > 0) {
        img_shared[0][tid_x + 1] = img[(y - 1) * width + x];
    }

    if (tid_y == 15 && y < height - 1) {
        img_shared[17][tid_x + 1] = img[(y + 1) * width + x];
    }

    if (tid_x == 0 && tid_y == 0 && x > 0 && y > 0) {
        img_shared[0][0] = img[(y - 1) * width + x - 1];
    }

    if (tid_x == 15 && tid_y == 0 && x < width - 1 && y > 0) {
        img_shared[0][17] = img[(y - 1) * width + x + 1];
    }

    if (tid_x == 0 && tid_y == 15 && x > 0 && y < height - 1) {
        img_shared[17][0] = img[(y + 1) * width + x - 1];
    }

    if (tid_x == 15 && tid_y == 15 && x < width - 1 && y < height - 1) {
        img_shared[17][17] = img[(y + 1) * width + x + 1];
    }

    __syncthreads();

    if (y > 0 && y < height - 1 && x > 0 && x < width - 1) {
        temp_f = 0.0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                temp_f += img_shared[tid_y + i + 1][tid_x + j + 1];
            }
        }

        temp_f /= 9;
        filter[y * width + x] = (unsigned char) temp_f;
    }
}

// 使用纹理存储
__global__ void mean_filter4(unsigned char *filter, cudaTextureObject_t texObj, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float temp_f;

    if (y > 0 && y < height - 1 && x > 0 && x < width - 1) { 
        temp_f = 0.0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                temp_f += tex2D<unsigned char>(texObj, (x + j + 0.5) / width, (y + i + 0.5) / height);
            }
        }
        temp_f /= 9;
        filter[y * width + x] = (unsigned char) temp_f;
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printDeviceProp(prop);

    Mat image = imread("lena_salt_pepper.bmp", IMREAD_GRAYSCALE);
    Mat dstImg0 = Mat(Size(image.cols, image.rows), CV_8UC1);
    Mat dstImg1 = Mat(Size(image.cols, image.rows), CV_8UC1);
    Mat dstImg2 = Mat(Size(image.cols, image.rows), CV_8UC1);
    Mat dstImg3 = Mat(Size(image.cols, image.rows), CV_8UC1);
    Mat dstImg4 = Mat(Size(image.cols, image.rows), CV_8UC1);

    int width = image.cols;
    int height = image.rows;

    unsigned char *d_in;
    unsigned char *d_out1;
    unsigned char *d_out2;
    unsigned char *d_out3;
    unsigned char *d_out4;
    cudaArray *srcArray;

    // serial part
    clock_t start0 = clock();
    serial_mean_filter(image.data, dstImg0.data, width, height);
    clock_t stop0 = clock();
    printf("serial filter: %f milliseconds\n", (double)(stop0 - start0) / 1000);
    
    size_t mem_size = width * height * sizeof(unsigned char);
    cudaSetDevice(2);

    //===================步骤1===================
    // Allocate memory on GPU
    cudaMalloc((void**)&d_in, mem_size);
	cudaMalloc((void**)&d_out1, mem_size);
    cudaMalloc((void**)&d_out2, mem_size);
    cudaMalloc((void**)&d_out3, mem_size);
    cudaMalloc((void**)&d_out4, mem_size);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaMallocArray(&srcArray, &channelDesc, width, height);

    //===================步骤2===================
    // copy operator to GPU
    cudaMemcpy(d_in, image.data, mem_size, cudaMemcpyHostToDevice);

    cudaMemcpy2DToArray(srcArray, 0, 0, image.data, width * sizeof(unsigned char), width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
    
    // texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = srcArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    //===================步骤3===================
    // GPU do the work, CPU waits
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    cudaEvent_t start, stop;
    float time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // one block multiple threads
    cudaEventRecord(start, 0);
    mean_filter1<<<1, blockSize>>>(d_in, d_out1, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("one block multiple threads: %f milliseconds\n", time);

    // multiple blocks multiple threads
    cudaEventRecord(start, 0);
    mean_filter2<<<gridSize, blockSize>>>(d_in, d_out2, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("multiple blocks multiple threads: %f milliseconds\n", time);

    // shared memory
    cudaEventRecord(start, 0);
    mean_filter3<<<gridSize, blockSize>>>(d_in, d_out3, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("shared memory: %f milliseconds\n", time);

    // texture memory
    cudaEventRecord(start, 0);
    mean_filter4<<<gridSize, blockSize>>>(d_out4, texObj, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("texture memory: %f milliseconds\n", time);

    //===================步骤4===================
    // Get results from the GPU
    cudaMemcpy(dstImg1.data, d_out1, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dstImg2.data, d_out2, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dstImg3.data, d_out3, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dstImg4.data, d_out4, mem_size, cudaMemcpyDeviceToHost);

    imwrite("result0.bmp", dstImg0);
    imwrite("result1.bmp", dstImg1);
    imwrite("result2.bmp", dstImg2);
    imwrite("result3.bmp", dstImg3);
    imwrite("result4.bmp", dstImg4);

    //===================步骤5===================
    // Free the memory
    cudaFree(d_in);
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_out3);
    cudaFree(d_out4);
    
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(srcArray);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaDeviceReset();
    return 0;
}