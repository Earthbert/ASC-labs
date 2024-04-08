#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

__device__ int pivot(int *v, int left, int right) {
	int pivotValue = v[left];

	while (left < right) {
		while (v[right] >= pivotValue && left < right) right--;
		v[left] = v[right];

		while (v[left] <= pivotValue && left < right) left++;
		v[right] = v[left];
	}

	v[left] = pivotValue;
	return left;
}

__global__ void quicksort(int *v, int left, int right, cudaStream_t parentStream) {
	if (left < right) {
		int pivotIndex = pivot(v, left, right); // Assume pivot is device function or inlined

		cudaStream_t leftStream, rightStream;
		cudaStreamCreateWithFlags(&leftStream, cudaStreamNonBlocking);
		cudaStreamCreateWithFlags(&rightStream, cudaStreamNonBlocking);

		if (left < pivotIndex) {
			quicksort << <1, 1, 0, leftStream >> > (v, left, pivotIndex - 1, leftStream);
		}
		if (pivotIndex + 1 < right) {
			quicksort << <1, 1, 0, rightStream >> > (v, pivotIndex + 1, right, rightStream);
		}

		cudaStreamDestroy(leftStream);
		cudaStreamDestroy(rightStream);
	}
}

int main() {
	int *v;  // Unified pointer
	int n;

	// Read from file
	std::ifstream fin("input.txt");
	if (!fin) {
		std::cerr << "Failed to open input file." << std::endl;
		return -1;
	}

	fin >> n;
	cudaMallocManaged(&v, n * sizeof(int));
	if (v == NULL)
		return -1;

	for (int i = 0; i < n; i++)
		fin >> v[i];
	fin.close();

	cudaStream_t stream;
	cudaStreamCreate(&stream); // Create a non-blocking stream

	// Call the quicksort kernel with the newly created stream
	quicksort << <1, 1, 0, stream >> > (v, 0, n - 1, stream);

	// Synchronize the stream to ensure sorting completes before proceeding
	cudaStreamSynchronize(stream);

	// Cleanup
	cudaStreamDestroy(stream);

	// Print sorted array
	for (int i = 0; i < n; i++)
		std::cout << v[i] << " ";
	std::cout << std::endl;

	cudaFree(v);

	return 0;
}
