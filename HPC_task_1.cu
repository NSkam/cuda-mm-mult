#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <thread>
#include <chrono>
#include <curand.h>
#pragma comment(lib,"cublas.lib")

using namespace std;

int lines, columns;

void init() {
	bool endc = false;
	bool endl = false;
	while (!endl || !endc)
	{
		printf("Give me the size of lines\n");

		scanf("%d", &lines);
		endl = true;

		if (lines <= 0) {
			printf("You must use a number of lines that is greater than 0!\n");
			endl = false;
		}
		if (endl == true) {
			printf("Give me the size of columns\n");
			scanf("%d", &columns);
			endc = true;
			if (columns <= 0) {
				printf("You must use a number of columns that is greater than 0!\n");
				endc = false;
				columns = 0;
			}
		}
	}
}


int gpu_blas_mmul(const float* A, float* C, const int row, const int col) {
	float time;
	cudaEvent_t start, stop;
	cublasStatus_t stat;
	cublasHandle_t handle;
	//CuBLAS initialization
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS initialization failed\n");
		return EXIT_FAILURE;
	}
		int lda = row, ldb=row, ldc = col;
		const float alf = 1;
		const float bet = 0;
	    const float* alpha = &alf;
	    const float* beta = &bet;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

     //The actual calculation
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, col, col, row, alpha, A, lda, A, ldb, beta, C, ldc);
	// for (int i = 0; i < (columns) * (columns); i++) printf("C[%d] = %.4f\n", i, C[i]);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("Time for calculations:  %3.6f ms \n", time);

	// Destroy the handle
	 cublasDestroy(handle);

}

//Fill the matrix with random numbers
 void GPU_fill_rand(float* A, int nr_rows_A, int nr_cols_A) {
     curandGenerator_t prng;
     curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
     curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
     curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

 //Print matrix
 void print_matrix(const float* A, int nr_rows_A, int nr_cols_A) { 
	 for (int i = 0; i < nr_rows_A; ++i) {
		 for (int j = 0; j < nr_cols_A; ++j) {
			 printf("C(%d,%d)=%.2f\n", i+1,j+1, A[j * nr_rows_A + i]);
		 }
	 }
 }


int main() {

	cudaError_t cudaStat;

	//time variables
	float time;
	cudaEvent_t start, stop;

	init();
	float* A = (float*)malloc(lines * columns * sizeof(float));
	float* C = (float*)malloc(columns * columns * sizeof(float));
	//For Matlab Testing
/*	for (int i = 0; i < lines; i++)
		for (int j = 0; j < columns; j++)
			if (j < columns)
				printf("A(%d,%d) = %.4f;\n", i + 1, j + 1, A[i * columns + j]);*/
	float* A_D;
	float* C_D;

	//Allocate memory in device for A
	cudaStat = cudaMalloc(&A_D, lines * columns * sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf("device memory allocation failed");
		return EXIT_FAILURE;
	}
	//Allocate memory in device for C
	cudaStat = cudaMalloc(&C_D, columns * columns * sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf("device memory allocation failed");
		return EXIT_FAILURE;
	}  

	GPU_fill_rand(A_D, lines, columns);

	cudaMemcpy(A, A_D, lines * columns * sizeof(float), cudaMemcpyDeviceToHost);
	//print_matrix(A, lines, columns);

	//Call multiply function
	gpu_blas_mmul(A_D, C_D, lines, columns);
	
	//print the time
	//printf("Time for calculations:  %3.1f ms \n", time);

	// Copy (and print) the result on host memory
	cudaMemcpy(C, C_D, (columns*columns * sizeof(float)), cudaMemcpyDeviceToHost);
	//print_matrix(C, columns, columns);
	cudaFree(A_D);
	cudaFree(C_D);
	free(A);
	free(C);
	return 0;
}