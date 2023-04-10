#include <iostream>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

typedef vector<vector<int> > Matrix;

void matrix_mul(vector<vector<int> > A, vector<vector<int> > B, vector<vector<int> > &C, int size);
void matrix_mul_tran(vector<vector<int> > A, vector<vector<int> > BT, vector<vector<int> > &C, int size);
void matrix_mul_para(vector<vector<int> > A, vector<vector<int> > B, vector<vector<int> > &C, int size);
void matrix_mul_para_tran(vector<vector<int> > A, vector<vector<int> > BT, vector<vector<int> >& C, int size);
bool matrix_check(vector<vector<int> > A, vector<vector<int> > B, int size);

int main()
{
	int size = 2048;
	vector<vector<int>> A(size, vector<int>(size));
	vector<vector<int>> B(size, vector<int>(size));
	vector<vector<int>> BT(size, vector<int>(size));
	vector<vector<int>> C(size, vector<int>(size));
	vector<vector<int>> C1(size, vector<int>(size));
	vector<vector<int>> C2(size, vector<int>(size));
	vector<vector<int>> C3(size, vector<int>(size));

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			A[i][j] = rand() % 100;
			B[i][j] = rand() % 100;
		}
	}
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			BT[i][j] = B[j][i];
		}
	}

	cout << "Computation Begin (serial)" << endl;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	matrix_mul(A, B, C, size);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	cout << "Time: " << static_cast <float> (duration) / 1000000.0f << endl << endl;
	

	cout << "Computation Begin (serial transposed)" << endl;
	t1 = high_resolution_clock::now();
	matrix_mul_tran(A, BT, C1, size);
	t2 = high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	cout << "Time: " << static_cast <float> (duration) / 1000000.0f << endl;
	cout << "Matirx Multiplication Computation Correctness: " << matrix_check(C, C1, size) << endl << endl;

	int thread_num;
	for (thread_num = 2; thread_num <= 16; thread_num += 2) {
		omp_set_num_threads(thread_num);
		cout << "Begin With Thread Num " << thread_num << endl;

		cout << "Computation Begin (parallel)" << endl;
		t1 = high_resolution_clock::now();
		matrix_mul_para(A, B, C2, size);
		t2 = high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Time: " << static_cast <float> (duration) / 1000000.0f << endl;
		cout << "Matirx Multiplication Computation Correctness: " << matrix_check(C, C2, size) << endl << endl;


		cout << "Computation Begin (parallel transposed)" << endl;
		t1 = high_resolution_clock::now();
		matrix_mul_para_tran(A, BT, C3, size);
		t2 = high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Time: " << static_cast <float> (duration) / 1000000.0f << endl;
		cout << "Matirx Multiplication Computation Correctness: " << matrix_check(C, C3, size) << endl << endl;

	}
	
	return 0;
}

void matrix_mul(vector<vector<int> > A, vector<vector<int> > B, vector<vector<int> > &C, int size) {
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k < size; k++)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void matrix_mul_tran(vector<vector<int> > A, vector<vector<int> > BT, vector<vector<int> >& C, int size) {
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k < size; k++)
			{
				C[i][j] += A[i][k] * BT[j][k];
			}
		}
	}
}

void matrix_mul_para(vector<vector<int> > A, vector<vector<int> > B, vector<vector<int> >& C, int size) {
	int i, j, k;
#pragma omp parallel shared(A, B, C, size) private(i, j, k)
	{
#pragma omp for schedule (static)
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				int sum = 0;
				for (k = 0; k < size; k++)
				{
					sum += A[i][k] * B[k][j];
				}
				C[i][j] = sum;
			}
		}
	}
}

void matrix_mul_para_tran(vector<vector<int> > A, vector<vector<int> > BT, vector<vector<int> >& C, int size) {
	int i, j, k;
#pragma omp parallel shared(A, BT, C, size) private(i, j, k)
	{
#pragma omp for schedule (static)
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				int sum = 0;
				for (k = 0; k < size; k++)
				{
					sum += A[i][k] * BT[j][k];
				}
				C[i][j] = sum;
			}
		}
	}
}

bool matrix_check(vector<vector<int> > A, vector<vector<int> > B, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (A[i][j] != B[i][j])
				return false;
		}
	}
	return true;
}