#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

void stdSort(std::vector<int>& v) {
	std::sort(v.begin(), v.end());
}

void mergeSort(std::vector<int>& v, unsigned long left, unsigned long right) {
	if (left < right) {
		unsigned long mid = left + (right - left) / 2;
		mergeSort(v, left, mid);
		mergeSort(v, mid + 1, right);
		std::inplace_merge(v.begin() + left, v.begin() + mid + 1, v.begin() + right + 1);
	}
}

void mergeSortParallel1(std::vector<int>& v, unsigned long left, unsigned long right) {
	if (left < right) {
		unsigned long mid = left + (right - left) / 2;
		#pragma omp taskgroup
		{
			#pragma omp task shared(v)
			mergeSort(v, left, mid);
			#pragma omp task shared(v)
			mergeSort(v, mid + 1, right);
			#pragma omp taskyield
		}
		std::inplace_merge(v.begin() + left, v.begin() + mid + 1, v.begin() + right + 1);
	}
}
void mm(std::vector<int>& v) {
#pragma omp parallel
#pragma omp single
	mergeSortParallel1(v, 0, v.size() - 1);
 }

int main() {
	std::cout << "my max threads number is " << omp_get_max_threads() << std::endl;
	
	std::vector<int> nums;
	for (int i = 16000000; i >= 0; i--)
		nums.push_back(i);
	std::vector<int> nums1(nums), nums2(nums);
	double start = omp_get_wtime();
	stdSort(nums);
	double end = omp_get_wtime();
	std::cout << "std sort time is: " << end - start << std::endl;

	start = omp_get_wtime();
	mergeSort(nums1, 0, nums1.size() - 1);
	end = omp_get_wtime();
	std::cout << "merge sort time is: " << end - start << std::endl;

	omp_set_num_threads(4);
	start = omp_get_wtime();
	mm(nums2);
	end = omp_get_wtime();
	std::cout << "parallel merge sort time is: " << end - start << std::endl;
	return 0;
}


/*int main() {
	const int N = 1000000;
	omp_set_num_threads(8);
	double sum = 0;
	int i;
	double start_p = omp_get_wtime();
#pragma omp parallel for reduction(+:sum)
	for (i = 0; i < 1000 * N; i++) {
		sum += 1 / (1 + ((double)(i) / N) * ((double)(i) / N)) / N;
	}
	double finish_p = omp_get_wtime();
	printf("sum: %lf\n", sum);
	printf("parallel time: %lf\n", finish_p - start_p);

	double sum_s = 0;
	double start = omp_get_wtime();
	for (i = 0; i < 1000 * N; i++) {
		sum_s += 1 / (1 + ((double)(i) / N) * ((double)(i) / N)) / N;
	}
	double finish = omp_get_wtime();
	printf("sum_s: %lf\n", sum_s);
	printf("serial time: %lf\n", finish - start);
	return 0;
}*/