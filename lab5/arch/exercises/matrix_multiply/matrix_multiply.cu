#include <stdlib.h>
#include <iostream>
#include <vector>

#define TRY_CUDA() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(-1); \
    }} while(0); \

#define TILE_WIDTH 16

__global__ void matrix_multiply_simple(float *ma, float *mb, float *mc, size_t width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float result = 0;
	for (int k = 0; k < width; ++k) {
		result += ma[row * width + k] * mb[k * width + col];
	}

	mc[row * width + col] = result;
}

__global__ void matrix_multiply(float *ma, float *mb, float *mc, size_t width) {
	__shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
	__shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float result = 0;

	for (int k = 0; k < width; k += TILE_WIDTH) {
		tile_a[threadIdx.y][threadIdx.x] = ma[row * width + k + threadIdx.x];
		tile_b[threadIdx.y][threadIdx.x] = mb[(k + threadIdx.y) * width + col];

		__syncthreads();

		for (int i = 0; i < TILE_WIDTH; ++i) {
			result += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
		}

		__syncthreads();
	}

	mc[row * width + col] = result;
}

int main(void) {
	// create a large workload so we can easily measure the
	// performance difference of both implementations

	// note that n measures the width of the matrix, not the number of total
	// elements
	const size_t n = 1 << 10;
	const dim3 block_size(TILE_WIDTH, TILE_WIDTH);
	const dim3 num_blocks(n / block_size.x, n / block_size.y);

	// generate random input on the host
	std::vector<float> host_a(n * n), host_b(n * n), host_c(n * n);
	for (int i = 0; i < n * n; ++i) {
		host_a[i] = static_cast<float>(rand()) / RAND_MAX;
		host_b[i] = static_cast<float>(rand()) / RAND_MAX;
	}

	// allocate storage for the device
	float *device_a = 0, *device_b = 0, *device_c = 0;
	cudaMalloc((void **)&device_a, sizeof(float) * n * n);
	cudaMalloc((void **)&device_b, sizeof(float) * n * n);
	cudaMalloc((void **)&device_c, sizeof(float) * n * n);

	// copy input to the device
	cudaMemcpy(device_a, &host_a[0], sizeof(float) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, &host_b[0], sizeof(float) * n * n, cudaMemcpyHostToDevice);

	cudaEvent_t launch_begin, launch_end;
	cudaEventCreate(&launch_begin);
	cudaEventCreate(&launch_end);

	// time many kernel launches and take the average time
	const size_t num_launches = 100;
	float average_simple_time = 0;
	std::cout << "Timing simple implementation..." << std::endl;

	for (int i = 0; i < num_launches; ++i) {
		cudaEventRecord(launch_begin);

		matrix_multiply_simple << <num_blocks, block_size >> > (device_a, device_b, device_c, n);

		cudaEventRecord(launch_end);
		cudaEventSynchronize(launch_end);

		float time;
		cudaEventElapsedTime(&time, launch_begin, launch_end);

		std::cout << "Iteration " << i << ": " << time << "ms" << std::endl;
		average_simple_time += time;
	}

	average_simple_time /= num_launches;
	std::cout << "Done." << std::endl;

	float average_optimized_time = 0;
	std::cout << "Timing optimized implementation..." << std::endl;

	for (int i = 0; i < num_launches; ++i) {
		cudaEventRecord(launch_begin);
		TRY_CUDA();

		matrix_multiply << <num_blocks, block_size >> > (device_a, device_b, device_c, n);
		TRY_CUDA();

		cudaEventRecord(launch_end);
		TRY_CUDA();
		cudaEventSynchronize(launch_end);
		TRY_CUDA();

		float time = 0;
		cudaEventElapsedTime(&time, launch_begin, launch_end);

		std::cout << "Iteration " << i << ": " << time << "ms" << std::endl;
		average_optimized_time += time;
	}

	average_optimized_time /= num_launches;
	std::cout << "Done." << std::endl;

	// report the effective throughput of each kernel in GFLOPS
	// the effective throughput is measured as the number of floating point
	// operations performed per second: (one mul + one add) * N^3
	float simple_throughput = (2 * n * n * n) / (average_simple_time / 1000.0f) / 1000000000.0f;
	float optimized_throughput = (2 * n * n * n) / (average_optimized_time / 1000.0f) / 1000000000.0f;

	std::cout << "Matrix size: " << n << "x" << n << std::endl;
	std::cout << "Tile size: " << TILE_WIDTH << "x" << TILE_WIDTH << std::endl;

	std::cout << "Throughput of simple kernel: " << simple_throughput << " GFLOPS" << std::endl;
	std::cout << "Throughput of optimized kernel: " << optimized_throughput << " GFLOPS" << std::endl;
	std::cout << "Performance improvement: " << optimized_throughput / simple_throughput << "x" << std::endl;
	std::cout << std::endl;

	cudaEventDestroy(launch_begin);
	cudaEventDestroy(launch_end);

	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);

	return 0;
}
