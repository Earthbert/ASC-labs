#include <stdio.h>
#include "utils.h"

#define N_BLOCKS 4
#define N_THREADS 4

/**
 * ~TODO 3~
 * Modify the kernel below such as each element of the
 * array will be now equal to 0 if it is an even number
 * or 1, if it is an odd number
 */
__global__ void kernel_parity_id(int *a, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        a[i] = i % 2;
    }
}

/**
 * ~TODO 4~
 * Modify the kernel below such as each element will
 * be equal to the BLOCK ID this computation takes
 * place.
 */
__global__ void kernel_block_id(int *a, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        a[i] = blockIdx.x;
    }
}

/**
 * ~TODO 5~
 * Modify the kernel below such as each element will
 * be equal to the THREAD ID this computation takes
 * place.
 */
__global__ void kernel_thread_id(int *a, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        a[i] = threadIdx.x;
    }
}

int main(void) {
    int nDevices;

    // Get the number of CUDA-capable GPU(s)
    cudaGetDeviceCount(&nDevices);

    /**
     * ~TODO 1~
     * For each device, show some details in the format below,
     * then set as active device the first one (assuming there
     * is at least CUDA-capable device). Pay attention to the
     * type of the fields in the cudaDeviceProp structure.
     *
     * Device number: <i>
     *      Device name: <name>
     *      Total memory: <mem>
     *      Memory Clock Rate (KHz): <mcr>
     *      Memory Bus Width (bits): <mbw>
     *
     * Hint: look for cudaGetDeviceProperties and cudaSetDevice in
     * the Cuda Toolkit Documentation.
     */
    for (int i = 0; i < nDevices; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("Device number: %d\n", i);
        printf("    Device name: %s\n", deviceProp.name);
        printf("    Total memory: %zu\n", deviceProp.totalGlobalMem);
        printf("    Memory Clock Rate (KHz): %d\n", deviceProp.memoryClockRate);
        printf("    Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);
    }


    /**
     * ~TODO 2~
     * With information from example_2.cu, allocate an array with
     * integers (where a[i] = i). Then, modify the three kernels
     * above and execute them using 4 blocks, each with 4 threads.
     *
     * You can use the fill_array(int *a, int n) function (from utils)
     * to fill your array as many times you want.
     *
     *  ~TODO 3~
     * Execute kernel_parity_id kernel and then copy from
     * the device to the host; call cudaDeviceSynchronize()
     * after a kernel execution for safety purposes.
     */
    int *host_array = (int *)malloc(N_BLOCKS * N_THREADS * sizeof(float));
    fill_array_int(host_array, N_BLOCKS * N_THREADS);

    int *device_array;
    cudaMalloc(&device_array, N_BLOCKS * N_THREADS * sizeof(int));
    TRY_CUDA();
    cudaMemcpy(device_array, host_array, N_BLOCKS * N_THREADS * sizeof(int), cudaMemcpyHostToDevice);
    TRY_CUDA();

    kernel_parity_id << <N_BLOCKS, N_THREADS >> > (device_array, N_BLOCKS * N_THREADS);
    TRY_CUDA();
    cudaMemcpy(host_array, device_array, N_BLOCKS * N_THREADS * sizeof(int), cudaMemcpyDeviceToHost);
    TRY_CUDA();

    check_task_3(3, host_array);

    /**
     * ~TODO 4~
     * Execute kernel_block_id kernel and then copy from
     * the device to the host;
     */
    kernel_block_id << <N_BLOCKS, N_THREADS >> > (device_array, N_BLOCKS * N_THREADS);
    cudaMemcpy(host_array, device_array, N_BLOCKS * N_THREADS * sizeof(int), cudaMemcpyDeviceToHost);

    check_task_3(4, host_array);

    /**
     * ~TODO 5~
     * Execute kernel_thread_id kernel and then copy from
     * the device to the host;
     */

     // Uncomment the line below to check your results
    kernel_thread_id << <N_BLOCKS, N_THREADS >> > (device_array, N_BLOCKS * N_THREADS);
    cudaMemcpy(host_array, device_array, N_BLOCKS * N_THREADS * sizeof(int), cudaMemcpyDeviceToHost);
    check_task_3(5, host_array);

    // TODO 6: Free the memory
    free(host_array);
    cudaFree(device_array);

    return 0;
}