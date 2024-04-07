#include <chrono>
#include <iostream>

// CUDA kernel for performing saxpy operation
__global__ void saxpy(int n, float a, float *x, float *y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) y[i] = a * x[i] + y[i];
}

// Function to get the current CPU timer in seconds
double myCPUTimer() {
	return static_cast<double>(std::chrono::high_resolution_clock::now()
		.time_since_epoch()
		.count()) /
		1e9;
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

	double t1, t2;

	// Start CPU timer
	t1 = myCPUTimer();

	// Launch saxpy kernel
	saxpy << <(N + 255) / 256, 256 >> > (N, 2.0f, d_x, d_y);

	// Wait for GPU to finish
	cudaDeviceSynchronize();

	// Stop CPU timer
	t2 = myCPUTimer();

	// Copy result back to host
	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	// Print the time in seconds
	std::cout << "Time using CPU timer: " << (t2 - t1) << " seconds."
		<< std::endl;

	// Cleanup
	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);

	return 0;
}
