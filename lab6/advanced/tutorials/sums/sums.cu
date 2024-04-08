#include <stdio.h>

#define NUM_ELEM 8
#define NUM_THREADS 10

__global__ void concurrentRW(int *data) {
	// NUM_THREADS try to read and write at same location

	atomicAdd(&data[blockIdx.x], threadIdx.x);
	// Equivalent to
	// data[blockIdx.x] = data[blockIdx.x] + threadIdx.x;
}

int main(int argc, char *argv[]) {
	int *data = NULL;
	bool errorsDetected = false;

	cudaMallocManaged(&data, NUM_ELEM * sizeof(*data));
	if (data == 0) {
		perror("[HOST] Couldn't allocate memory");
		return 1;
	}

	// init all elements to 0
	cudaMemset(data, 0, NUM_ELEM);

	// launch kernel writes
	concurrentRW << <NUM_ELEM, NUM_THREADS >> > (data);
	cudaDeviceSynchronize();
	if (cudaSuccess != cudaGetLastError()) {
		return 1;
	}

	for (int i = 0; i < NUM_ELEM; i++) {
		printf("%d. %d\n", i, data[i]);
		if (data[i] != (NUM_THREADS * (NUM_THREADS - 1) / 2)) {
			errorsDetected = true;
		}
	}

	if (errorsDetected) {
		perror("Errors detected");
	} else {
		puts("OK");
	}

	return 0;
}
