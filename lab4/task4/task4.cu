#include <stdio.h>
#include <math.h>
#include "utils.h"

__global__ void add_arrays(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    cudaSetDevice(0);
    int N = 1 << 20;

    float* host_array_a = 0;
    float* host_array_b = 0;
    float* host_array_c = 0;

    float* device_array_a = 0;
    float* device_array_b = 0;
    float* device_array_c = 0;

    host_array_a = (float*)malloc(N * sizeof(float));
    host_array_b = (float*)malloc(N * sizeof(float));
    host_array_c = (float*)malloc(N * sizeof(float));

    cudaMalloc(&device_array_a, N * sizeof(float));
    cudaMalloc(&device_array_b, N * sizeof(float));
    cudaMalloc(&device_array_c, N * sizeof(float));

    if (host_array_a == 0 || host_array_b == 0 || host_array_c == 0 ||
        device_array_a == 0 || device_array_b == 0 || device_array_c == 0) {
        printf("[ERROR] Memory allocation failed\n");
        return 1;
    }

    fill_array_float(host_array_a, N);
    fill_array_random(host_array_b, N);

    cudaMemcpy(device_array_a, host_array_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_array_b, host_array_b, N * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    add_arrays<<<grid_size, block_size>>>(device_array_a, device_array_b, device_array_c, N);

    cudaMemcpy(host_array_c, device_array_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    check_task_4(host_array_a, host_array_b, host_array_c, N);

    free(host_array_a);
    free(host_array_b);
    free(host_array_c);
    cudaFree(device_array_a);
    cudaFree(device_array_b);
    cudaFree(device_array_c);
    
    return 0;
}