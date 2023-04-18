#include "cuda_runtime.h"
#include <iostream>
#include "util.h"

using namespace std;

template<typename T>
class Complex {
public:
    T real{};
    T imag{};

    constexpr static const double PI = 3.1415926536;

    Complex() = default;

    static Complex getComplex(T real, T imag) {
        Complex res;
        res.real = real;
        res.imag = imag;
        return res;
    }

    __device__ static Complex W(int n, int k) {
        return Complex(cos(-2.0 * PI * k / n), sin(-2.0 * PI * k / n));
    }

    __device__ Complex(T real, T imag) {
        this->real = real;
        this->imag = imag;
    }

    __device__ Complex operator+(const Complex &rhs) {
        return Complex(this->real + rhs.real, this->imag + rhs.imag);
    }

    __device__ Complex operator-(const Complex &rhs) {
        return Complex(this->real - rhs.real, this->imag - rhs.imag);
    }

    __device__ Complex operator*(const Complex &rhs) {
        return Complex(this->real * rhs.real - this->imag * rhs.imag,
                       this->imag * rhs.real + this->real * rhs.imag);
    }
};

template<typename T>
ostream &operator<<(ostream &os, const Complex<T> rhs) {
    os << "(" << rhs.real << ", " << rhs.imag << ")";
    return os;
}

__device__ uint cubitReverse(uint x, int n) {
    uint res = 0;
    while (n >>= 1) {
        res <<= 1;
        res |= (x & 1);
        x >>= 1;
    }
    return res;
}

template<typename T>
__device__ void butterfly(Complex<T> &a, Complex<T> &b, Complex<T> &factor) {
    Complex<T> f1 = a + factor * b;
    Complex<T> f2 = a - factor * b;
    a = f1, b = f2;
}

template<typename T>
__global__ void init(Complex<T> signal[], Complex<T> transform[], int n) {
    uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (uint i = tid; i < SIGNAL_SIZE; i += STRIDE) {
        transform[cubitReverse(i, n)] = signal[i];
    }
}

template<typename T>
__global__ void step(Complex<T> transform[], int m, int mask) {
    uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint offset = m >> 1;
    for (uint i = tid; i < PROCESSOR_SIZE; i += STRIDE) {
        uint k = i & mask;
        uint index = ((i & ~mask) << 1) | k;
        Complex<T> factor = Complex<T>::W(m, k);
        butterfly(transform[index], transform[index + offset], factor);
    }
}

template<typename T>
void FFT(Complex<T> signal[], Complex<T> transform[], int n) {
    init<T><<<BLOCKS_NUM, THREADS_NUM>>>(signal, transform, n);
    for (int m = 2, mask = 0; m <= n; m <<= 1, mask = (mask << 1) | 1) {
        step<T><<<BLOCKS_NUM, THREADS_NUM>>>(transform, m, mask);
    }
}


