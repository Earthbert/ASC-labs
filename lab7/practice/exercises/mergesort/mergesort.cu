#include <stdio.h>

#define NUM_ELEMENTS 16

__device__ void merge(int *arr, int *temp, int left, int middle, int right) {
	int i = left;
	int j = middle;
	int k = left;

	while (i < middle && j < right) {
		// TODO 1: Get the smaller element
		if(arr[i] < arr[j])
			temp[k++] = arr[i++];
		else
			temp[k++] = arr[j++];
	}

	while (i < middle) {
		// TODO 2: Copy any remaining elements from the left subarray
		temp[k++] = arr[i++];
	}
	while (j < right) {
		// TODO 3: Copy any remaining elements from the right subarray
		temp[k++] = arr[j++];
	}

	for (int x = left; x < right; x++) {
		// TODO 4: Copy the sorted elements
		arr[x] = temp[x];
	}
}

__global__ void mergeSortGPU(int *arr, int *temp, int n, int width) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int left = tid * width;
	int middle = left + width / 2;
	int right = left + width;

	if (left < n && middle < n) {
		merge(arr, temp, left, middle, right);
	}
}

void generateData(int *data, int size) {
	srand(time(0));

	for (int i = 0; i < size; i++) {
		data[i] = rand() % 14 + 1;
	}
}

int main() {
	int *array = NULL;
	array = (int *)malloc(NUM_ELEMENTS * sizeof(int));
	generateData(array, NUM_ELEMENTS);

	printf("Original Array: ");
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d ", array[i]);
	}
	printf("\n");

	int *d_array, *d_tmp;
	cudaMalloc(&d_array, NUM_ELEMENTS * sizeof(int));
	cudaMemcpy(d_array, array, NUM_ELEMENTS * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMalloc(&d_tmp, NUM_ELEMENTS * sizeof(int));

	int block_size = 256;
	size_t blocks_no = NUM_ELEMENTS / block_size;

	if (NUM_ELEMENTS % block_size) ++blocks_no;

	for (int width = 1; width < NUM_ELEMENTS; width *= 2) {
		// TODO 0: Call the mergeSortGPU kernel with the appropriate arguments
		mergeSortGPU << <blocks_no, block_size >> > (d_array, d_tmp, NUM_ELEMENTS, width);
	}
	cudaDeviceSynchronize();

	cudaMemcpy(array, d_array, NUM_ELEMENTS * sizeof(int),
		cudaMemcpyDeviceToHost);
	cudaFree(d_array);

	printf("Sorted Array: ");
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d ", array[i]);
	}
	printf("\n");

	return 0;
}
