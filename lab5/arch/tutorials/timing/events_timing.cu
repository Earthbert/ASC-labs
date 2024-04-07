#include <iostream>

// CUDA kernel for performing saxpy operation
__global__ void saxpy(int n, float a, float *x, float *y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
	int N = 1 << 20;  // Example size of vectors
	float *x, *y, *d_x, *d_y;

	// Allocate host memory
	x = (float *)malloc(N * sizeof(float));
	y = (float *)malloc(N * sizeof(float));

	// Initialize x and y arrays
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// Allocate device memory
	cudaMalloc(&d_x, N * sizeof(float));
	cudaMalloc(&d_y, N * sizeof(float));

	// Copy inputs to device
	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record the start event
	cudaEventRecord(start);

	// Launch saxpy kernel
	saxpy << <(N + 255) / 256, 256 >> > (N, 2.0f, d_x, d_y);

	// Record the stop event
	cudaEventRecord(stop);

	// Copy result back to host
	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// Convert milliseconds to seconds and print
	std::cout << "Time using CUDA events: " << milliseconds / 1000.0f
		<< " seconds." << std::endl;

	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);

	return 0;
}
