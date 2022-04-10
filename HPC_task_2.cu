#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <device_functions.h>

#define TILE_WIDTH 2

using namespace std;

int lines, columns;

void init() {
	bool endl = false;
	bool endc = false;
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

static void table_generator(float* random_table) {

	srand(time(NULL));
	for (int i = 0; i < lines * columns; i++) {
		srand(time(NULL) * rand());
		random_table[i] = (float)rand() / 53;
	}
}

__global__ void kernel(float* A, float* C, int l, int c) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x < c && y < c) {
		printf("coordinates (x,y) = (%d,%d)\n", x, y);;
		C[y * c + x] = 0;
		for (int i = 0; i < l; i++) {
			//printf("loop = %d and coordinates (x,y) = (%d,%d)\n",i , x, y);
			C[y * c + x] = A[i * c + y] * A[i * c + x] + C[y * c + x];
		}
	}
}


int main() {

	float time;
	cudaEvent_t start, stop;

	init();
	float* A = (float*)malloc((lines * columns) * sizeof(float));
	float* C = (float*)malloc((columns * columns) * sizeof(float));


	table_generator(A);
	//For Matlab Testing
	//for (int i = 0; i < lines; i++) 
	//	for (int j = 0; j < columns; j++)
	//		printf("A(%d,%d) = %.4f;\n", i + 1, j + 1, A[i*columns + j]);
	float* A_D;
	float* C_D;
	cudaMalloc((void**)&A_D, lines * columns * sizeof(float));
	cudaMalloc((void**)&C_D, columns * columns * sizeof(float));

	cudaMemcpy(A_D, A, lines * columns * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C_D, C, columns * columns * sizeof(float), cudaMemcpyHostToDevice);

	float size = (float)columns / (float)TILE_WIDTH;
	if (size != (int)size) size = (int)size + 1;
	dim3 dimBlock(TILE_WIDTH , TILE_WIDTH, 1);
	dim3 dimGrid(size, size, 1);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	kernel <<< dimGrid, dimBlock >>> (A_D, C_D, lines, columns);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("Time for calculations:  %3.6f ms \n", time);

	cudaMemcpy(C, C_D, columns * columns * sizeof(float), cudaMemcpyDeviceToHost);
	//for (int i= 0; i < columns * columns; i++) printf("C[%d] = %.4f\n", i, C[i]);
	cudaFree(A_D);
	cudaFree(C_D);
	free(A);
	free(C);
	return 0;
}