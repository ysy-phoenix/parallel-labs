#include <iostream>
#include <random>
#include <algorithm>
#include "omp.h"

using namespace std;

// 定义取样的数量和线程数量
const int SAMPLE_SIZE = 4000;
const int THREADS_NUM = 8;
const int SIZE = 150000000;
#define TASK_SIZE 800

// 使用随机设备生成随机数引擎
std::random_device rd;
std::mt19937 gen(rd());
int nums[SIZE];
int nums2[SIZE];
int temp[SIZE];

// 初始化数据
void initArray(int arr1[], int arr2[]) {
    std::uniform_int_distribution<> dist(0, SIZE);
    for (int i = 0; i < SIZE; ++i) {
        int num = dist(gen) + 1;
        arr1[i] = num;
        arr2[i] = num;
    }
}

bool check(const int arr[]) {
    for (int i = 0; i < SIZE - 1; ++i) {
        if (arr[i] > arr[i + 1]) {
            cout << i << " " << arr[i] << endl;
            return false;
        }
    }
    return true;
}

void merge(int arr[], int low, int mid, int high) {
    int l = low, r = mid + 1, index = low;
    while (l <= mid && r <= high) {
        if (arr[l] < arr[r]) {
            temp[index++] = arr[l++];
        } else {
            temp[index++] = arr[r++];
        }
    }
    while (l <= mid) {
        temp[index++] = arr[l++];
    }
    while (r <= high) {
        temp[index++] = arr[r++];
    }
    for (int i = low; i <= high; ++i) {
        arr[i] = temp[i];
    }
}

void mergeSort(int arr[], int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;
        mergeSort(arr, low, mid);
        mergeSort(arr, mid + 1, high);
        merge(arr, low, mid, high);
    }
}

void parallelMergeSort(int arr[], int low, int high) {
    int n = high - low + 1;
    if (n < 2) {
        return;
    }

    int mid = (low + high) / 2;

    #pragma omp task default(none) shared(arr, low, mid) if (n > TASK_SIZE)
    parallelMergeSort(arr, low, mid);

    #pragma omp task default(none) shared(arr, mid, high) if (n > TASK_SIZE)
    parallelMergeSort(arr, mid + 1, high);

    #pragma omp taskwait
    merge(arr, low, mid, high);
}

void parallelMergeSortEntry(int arr[], int low, int high) {
    #pragma omp parallel default (none) shared(arr, low, high)
    {
        #pragma omp single
        parallelMergeSort(arr, low, high);
    }
}

void selectPivot(int arr[], int low, int high) {
    // 设置随机数的范围和分布
    std::uniform_int_distribution<> dist(low, high);
    int pivotIndex = dist(gen) % (high - low + 1) + low;
    swap(arr[pivotIndex], arr[high]);
}

int partition(int arr[], int low, int high) {
    selectPivot(arr, low, high);
    int pivot = arr[high], i = low - 1;
    for (int j = low; j <= high - 1; ++j) {
        if (arr[j] < pivot) {
            swap(arr[++i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int mid = partition(arr, low, high);
        quickSort(arr, low, mid - 1);
        quickSort(arr, mid + 1, high);
    }
}

void computeSamples(const int arr[], int samples[], int pivots[]) {
    const int step = (int) (SIZE / (SAMPLE_SIZE * THREADS_NUM));
    #pragma omp parallel num_threads(THREADS_NUM) default(none) shared(arr, samples)
    {
        int tid = omp_get_thread_num();
        int offset = tid * SAMPLE_SIZE * step;
        int index = tid * SAMPLE_SIZE;
        for (int i = 0; i < SAMPLE_SIZE; ++i) {
            samples[index + i] = arr[offset + (i * step)];
        }
    }
    quickSort(samples, 0, THREADS_NUM * SAMPLE_SIZE - 1);
    for (int i = 0; i < THREADS_NUM - 1; ++i) {
        pivots[i] = samples[(i + 1) * SAMPLE_SIZE];
    }
}

void partitionData(int arr[], const int pivots[], int partitions[], int start, int end) {
    for (int i = 1; i <= THREADS_NUM; ++i) {
        partitions[i] = end;
    }
    partitions[0] = start;

    int index = 1;
    for (int i = 0; i < THREADS_NUM - 1; ++i) {
        int lower = partitions[i], upper = partitions[i + 1], pivot = pivots[i];
        while (lower <= upper) {
            while (arr[lower] < pivot) {
                ++lower;
            }
            while (arr[upper] >= pivot) {
                --upper;
            }
            if (lower <= upper) {
                swap(arr[lower++], arr[upper--]);
            }
        }
        partitions[index++] = lower;
    }
}

void parallelQuickSort(int arr[], int low, int high) {
    int samples[SAMPLE_SIZE * THREADS_NUM];
    int partitions[THREADS_NUM + 1];
    int pivots[THREADS_NUM - 1];

    computeSamples(arr, samples, pivots);
    partitionData(arr, pivots, partitions, low, high);

    #pragma omp parallel num_threads(THREADS_NUM) default(none) shared(arr, partitions, high)
    {
        int tid = omp_get_thread_num();
        int left = partitions[tid], right = (tid == THREADS_NUM - 1) ? high : partitions[tid + 1] - 1;
        quickSort(arr, left, right);
    }
}

void testSort(int arr[], void(*sort)(int *, int, int), bool isSerial) {
    double start = omp_get_wtime();
    if (isSerial) {
        sort(arr, 0, SIZE - 1);
    } else {
        omp_set_dynamic(0);
        omp_set_num_threads(THREADS_NUM);
        sort(arr, 0, SIZE - 1);
    }
    double end = omp_get_wtime();

    bool flag = check(arr);
    if (flag) {
        if (isSerial) {
            std::cout << "Serial time: " << end - start << std::endl;
        } else {
            std::cout << "Parallel time: " << end - start << std::endl;
        }
    } else {
        std::cout << "Error!" << std::endl;
    }
}

void testMergeSort() {
    initArray(nums, nums2);
    testSort(nums, mergeSort, true);
    testSort(nums2, parallelMergeSortEntry, false);
}


void testQuickSort() {
    initArray(nums, nums2);
    testSort(nums, quickSort, true);
    testSort(nums2, parallelQuickSort, false);
}

int main() {
    testMergeSort();
//    testQuickSort();
    return 0;
}

/*
 * thread           2           4           6           8           10          12
 * merge serial     32.0976     31.8797     34.7177     32.4248     31.9894     32.1153
 * merge parallel   34.9321     18.9337     17.0003     13.4003     12.5569     13.1498
 * quick serial     32.5225     33.5187     31.9018     32.3487     31.6100     32.4221
 * quick parallel   17.4009     9.89134     8.6304      7.62502     6.84606     7.60903
 *
 * task size        25          50          100         200         400         800
 * merge serial     31.7625     32.2101     31.8397     31.8305     31.8176     31.9376
 * merge parallel   15.2853     15.0268     13.5111     14.9121     14.1888     15.025
 * */