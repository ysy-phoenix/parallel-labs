#include "util.h"

#include <complex>
#include <iostream>

using namespace std;


uint bitReverse(uint x, int n) {
    uint res = 0;
    while (n >>= 1) {
        res <<= 1;
        res |= (x & 1);
        x >>= 1;
    }
    return res;
}

template<typename T>
void fft(complex<T> signal[], complex<T> transform[], int n) {
    const complex<T> J(0, 1);
    // First bit reverse to find index.
    for (uint i = 0; i < n; ++i) {
        transform[bitReverse(i, n)] = signal[i];
    }
    // Iterate through each stage
    for (int m = 2; m <= n; m <<= 1) {
        int offset = m >> 1;
        complex<T> w(1, 0);
        complex<T> wm = exp(-J * (PI / offset));
        for (int j = 0; j < offset; ++j) {
            for (int k = j; k < n; k += m) {
                complex<T> t = w * transform[k + offset];
                complex<T> u = transform[k];
                transform[k] = u + t;
                transform[k + offset] = u - t;
            }
            w *= wm;
        }
    }
}