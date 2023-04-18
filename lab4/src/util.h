#ifndef FFT_UTIL_H_
#define FFT_UTIL_H_

const double PI = 3.1415926536;
using uint = unsigned int;
using dtype = double;

// Define the size of the serial_signal signal
const int N0 = 10;
const int N1 = 12;
const int STRIDE = 1 << (N0 + N1);
const int N = N0 + N1 + 3;
const int SIGNAL_SIZE = 1 << N;
const int PROCESSOR_SIZE = SIGNAL_SIZE >> 1;
const int THREADS_NUM = 1 << N0;
const int BLOCKS_NUM = 1 << N1;


#endif // FFT_UTIL_H_

/*
 *                              2^21            2^22            2^23            2^24            2^25
 *      cufft               4.23728ms       8.15056ms       15.7385ms       30.6392ms       59.814ms
 *      parallel FFT        24.2675ms       50.1064ms       104.178ms       155.305ms       256.189ms
 *      Serial FFT          4329ms          10718ms         21395ms         43404ms         98161ms
 *
 *                         1024 * 256       1024 * 512      1024 * 1024     2048 * 1024     4096 * 1024
 *      parallel FFT       250.83 ms        247.589 ms      250.156ms       248.511ms       247.877ms
 * */