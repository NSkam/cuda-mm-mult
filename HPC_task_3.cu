#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <thread>
#include <chrono>


#define TILE_WIDTH 25

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

static void table_generator(float* random_table) {

	srand(time(NULL));
	for (int i = 0; i < lines * columns; i++) {
		srand(time(NULL) * rand());
		random_table[i] = (float)rand() / 53;
	}
}

__global__ void kernel(float* A, float* C, int l, int c, int padding_offset) {

	//compute the columns without the padding
	int c_columns = c - padding_offset;
	//thread registers
	int by = blockIdx.y;
	int bx = blockIdx.x;

	int ty = threadIdx.y;
	int tx = threadIdx.x;
	//Move a tile from A to shared
	int col = bx * TILE_WIDTH + tx;
	int row = by * TILE_WIDTH + ty;
	//Shared_Memory
	if (col < c_columns) {
		__shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
		__shared__ float A_shared_inv[TILE_WIDTH][TILE_WIDTH];

		//Dont calculate padding rows/columns	
		for (int i = 0; i < l; i++) {

			//printf("This is block %d,%d and thread %d, %d\n", bx, by, tx, ty);
			A_shared[ty][tx] = A[i * c + col];
			A_shared_inv[ty][tx] = A[i * c + row];
			//printf("Memories A_shared[%d][%d] = %f and A_inv[%d][%d] = %f  \n We have block (%d,%d) where thread (%d,%d) is working right now!\n", ty, tx, A_shared[ty][tx], ty, tx, A_shared_inv[ty][tx], bx, by, tx, ty);

			if (bx == by) C[row * c_columns + col] += A_shared[ty][tx] * A_shared_inv[ty][tx];
			else if (bx > by) {
				C[row * c_columns + col] += A_shared[ty][tx] * A_shared_inv[ty][tx];
				C[col * c_columns + row] += A_shared[ty][tx] * A_shared_inv[ty][tx];
			}

		}

	}
}


int main() {
	
	float time;
	cudaEvent_t start, stop;

	init();
	int padding_offset = 0;
	if ((columns % 128) != 0) {
		padding_offset = 128 - (columns % 128);
	}
	//Collumn Padding
	columns += padding_offset;
	float* A = (float*)malloc((lines * columns) * sizeof(float));
	float* C = (float*)malloc((columns - padding_offset) * (columns - padding_offset) * sizeof(float));
	table_generator(A);
	//For Matlab Testing
	/*for (int i = 0; i < lines; i++)
		for (int j = 0; j < columns; j++)
			if (j < columns - padding_offset)
				printf("A(%d,%d) = %.4f;\n", i + 1, j + 1, A[i * columns + j]);*/
	float* A_D;
	float* C_D;
	cudaMalloc((void**)&A_D, lines * columns * sizeof(float));
	cudaMalloc((void**)&C_D, (columns - padding_offset) * (columns - padding_offset) * sizeof(float));

	cudaMemcpy(A_D, A, lines * columns * sizeof(float), cudaMemcpyHostToDevice);

	float size = (float)(columns - padding_offset) / (float)TILE_WIDTH;
	if (size != (int)size) size = (int)size + 1;
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 dimGrid(int(size), int(size), 1);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	kernel << < dimGrid, dimBlock >> > (A_D, C_D, lines, columns, padding_offset);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("Time for calculations:  %3.6f ms \n", time);

	cudaMemcpy(C, C_D, ((columns - padding_offset) * (columns - padding_offset) * sizeof(float)), cudaMemcpyDeviceToHost);

	//print C
	//for (int i = 0; i < (columns - padding_offset) * (columns - padding_offset); i++) printf("C[%d] = %.4f\n", i, C[i]);
	cudaFree(A_D);
	cudaFree(C_D);
	free(A);
	free(C);
	return 0;
}