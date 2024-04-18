#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEMENTS (1 << 20)

__device__ void swap(int *a, int *b) {
	int temp = *a;
	*a = *b;
	*b = temp;
}

// TODO 2: define parameters
__global__ void oddEvenTranspositionSort(int *array, int n) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < n; i++) {
		if (i % 2 == 0) {  // Even phase
			// TODO 2: Compare and swap elements if thread id is even
			if(tid % 2 == 0 && tid + 1 < n)
				if(array[tid] > array[tid + 1])
					swap(&array[tid], &array[tid + 1]);
		} else {           // Odd phase
			// TODO 3: Compare and swap elements if thread id is odd
			if(tid % 2 != 0 && tid + 1 < n)
				if(array[tid] > array[tid + 1])
					swap(&array[tid], &array[tid + 1]);
		}
		// TODO 4: Sync threads
		__syncthreads();
	}
}

void generateData(int *data, int size) {
	srand(time(0));

	for (int i = 0; i < size; i++) {
		data[i] = rand() % NUM_ELEMENTS + 1;
	}
}

int main() {
	int *array = NULL;
	array = (int *)malloc(NUM_ELEMENTS * sizeof(int));
	generateData(array, NUM_ELEMENTS);

	// printf("Original Array: ");
	// for (int i = 0; i < 10; i++) {
	// 	printf("%d ", array[i]);
	// }
	// printf("\n");

	time_t start, end;

	time(&start);
	int *d_array;
	// TODO 0: Allocate device array and copy host elements to it
	cudaMalloc(&d_array, NUM_ELEMENTS * sizeof(int));
	cudaMemcpy(d_array, array, NUM_ELEMENTS * sizeof(int),
		cudaMemcpyHostToDevice);

	// TODO 1: Calculate blocks_no and block_size
	int block_size = 128;
	int blocks_no = (NUM_ELEMENTS + block_size - 1) / block_size;
	oddEvenTranspositionSort << <blocks_no, block_size >> > (d_array, NUM_ELEMENTS);
	cudaDeviceSynchronize();

	cudaMemcpy(array, d_array, NUM_ELEMENTS * sizeof(int),
		cudaMemcpyDeviceToHost);
	cudaFree(d_array);

	time(&end);

	double time_taken = (double)(end - start); 
	printf("Time taken gpu: %f\n", time_taken);


	// printf("Sorted Array: ");
	// for (int i = 0; i < 10; i++) {
	// 	printf("%d ", array[i]);
	// }
	// printf("\n");

	free(array);
	return 0;
}
// 7 ... 3800 ... x
// 700 ... 16 ... y