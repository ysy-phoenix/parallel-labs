#include "util.h"
#include "serial_fft.cpp"
#include "parallel_fft.cu"
#include "cudafft.cu"

#include <iostream>
#include <random>
#include <chrono>
#include <complex>

using namespace std;

random_device rd;
mt19937 gen(rd());

// Create the serial_signal signal on the host
cutype *cu_signal, *cu_transform;
complex<dtype> *serial_signal, *serial_transform;
Complex<dtype> *parallel_signal, *parallel_transform;


void initMalloc() {
    cu_signal = (cutype *) malloc(sizeof(cutype) * SIGNAL_SIZE);
    cu_transform = (cutype *) malloc(sizeof(cutype) * SIGNAL_SIZE);
    serial_signal = (complex<dtype> *) malloc(sizeof(complex<dtype>) * SIGNAL_SIZE);
    serial_transform = (complex<dtype> *) malloc(sizeof(complex<dtype>) * SIGNAL_SIZE);
    parallel_signal = (Complex<dtype> *) malloc(sizeof(Complex<dtype>) * SIGNAL_SIZE);
    parallel_transform = (Complex<dtype> *) malloc(sizeof(Complex<dtype>) * SIGNAL_SIZE);
}

void freeAll() {
    free(cu_signal);
    free(cu_transform);
    free(serial_signal);
    free(serial_transform);
    free(parallel_signal);
    free(parallel_transform);
}

template<typename T>
void check(cutype res0[], Complex<T> res1[], complex<T> res2[]) {
    bool flag = true;
    for (int i = 0; i < SIGNAL_SIZE; ++i) {
        if ((abs(res0[i].x - res1[i].real) > 1e-6) || (abs(res0[i].y - res1[i].imag) > 1e-6)) {
            cout << "Parallel FFT Error at " << i << " expect " << res0[i] << " get " << res1[i] << endl;
            flag = false;
        } else if ((abs(res0[i].x - res2[i].real()) > 1e-5) || (abs(res0[i].y - res2[i].imag()) > 1e-5)) {
            cout << "Serial FFT Error at " << i << " expect " << res0[i] << " get " << res2[i] << endl;
            flag = false;
        }
    }
    if (flag) {
        cout << "success!" << endl;
    }
}

template<typename T>
void randomInit(cutype cuSignal[], complex<T> serialSignal[], Complex<dtype> parallelSignal[]) {
    uniform_real_distribution<> dis(0, 1);
    for (int i = 0; i < SIGNAL_SIZE; ++i) {
        auto x = dis(gen), y = dis(gen);
        cuSignal[i].x = x, cuSignal[i].y = y;
        serialSignal[i] = complex<T>(x, y);
        parallelSignal[i] = Complex<T>::getComplex(x, y);
    }
}

void parallel_fft() {
    Complex<dtype> *devInputSignal, *devOutputSignal;
    cudaMalloc(&devInputSignal, sizeof(complex<dtype>) * SIGNAL_SIZE);
    cudaMalloc(&devOutputSignal, sizeof(complex<dtype>) * SIGNAL_SIZE);

    cudaMemcpy(devInputSignal, parallel_signal, sizeof(complex<dtype>) * SIGNAL_SIZE, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    auto time = 0.f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, nullptr);
    FFT<dtype>(devInputSignal, devOutputSignal, SIGNAL_SIZE);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "Parallel FFT: " << time << "ms" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(parallel_transform, devOutputSignal, sizeof(complex<dtype>) * SIGNAL_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(devInputSignal);
    cudaFree(devOutputSignal);
}

void cufft() {
    // Allocate memory for the serial_signal and serial_transform signals on the device
    cutype *devInputSignal, *devOutputSignal;
    cudaMalloc(&devInputSignal, sizeof(cutype) * SIGNAL_SIZE);
    cudaMalloc(&devOutputSignal, sizeof(cutype) * SIGNAL_SIZE);

    // Copy the serial_signal signal from the host to the device
    cudaMemcpy(devInputSignal, cu_signal, sizeof(cutype) * SIGNAL_SIZE, cudaMemcpyHostToDevice);

    // Create a CUDA FFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_Z2Z, 1);

    cudaEvent_t start, stop;
    auto time = 0.f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Execute the FFT
    cudaEventRecord(start, nullptr);
    cufftExecZ2Z(plan, devInputSignal, devOutputSignal, CUFFT_FORWARD);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "CUFFT: " << time << "ms" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy the serial_transform signal from the device to the host
    cudaMemcpy(cu_transform, devOutputSignal, sizeof(cutype) * SIGNAL_SIZE, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(devInputSignal);
    cudaFree(devOutputSignal);
    cufftDestroy(plan);
}

void fftTest() {
    initMalloc();
    randomInit(cu_signal, serial_signal, parallel_signal); // cu_signal

    // cuda FFT
    cufft();

    // parallel FFT
    parallel_fft();

    // Serial FFT
    auto begin = chrono::high_resolution_clock::now();
    fft<dtype>(serial_signal, serial_transform, SIGNAL_SIZE);
    auto end = chrono::high_resolution_clock::now();
    cout << "Serial FFT: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "ms" << endl;

    // check result
    check<dtype>(cu_transform, parallel_transform, serial_transform);

    freeAll();
}


int main() {
    fftTest();
    return 0;
}
