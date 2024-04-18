#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define KERNEL_SIZE 5
#define KERNEL_RADIUS (KERNEL_SIZE / 2)
#define BLOCK_SIZE 16
#define WIDTH 512
#define HEIGHT 512

#define PI 3.14159265358979323846

// Gaussian Kernel (fixed size for simplicity)
__constant__ float d_kernel[KERNEL_SIZE];

// TODO
__global__ void gaussianBlur(const unsigned char *input, unsigned char *output,
	int width, int height) {
	// TODO 1: Calculate x and y
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int tid = y * width + x;

	float sum = 0.0;
	float weightSum = 0.0;

	for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
		for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
			int dX = x + kx;
			int dY = y + ky;
			if (dX >= 0 && dX < width && dY >= 0 && dY < height) {
				float weight = d_kernel[ky + KERNEL_RADIUS] * d_kernel[kx + KERNEL_RADIUS];
				sum += weight * input[dY * width + dX];
				weightSum += weight;
			}
		}
	}

	output[tid] = sum / weightSum;
}

void setGaussianKernel(float sigma) {
	float sum = 0.0;
	float s = 2.0 * sigma * sigma;
	float kernel[KERNEL_SIZE];

	for (int i = 0; i < KERNEL_SIZE; i++) {
		int x = i - KERNEL_RADIUS;
		kernel[i] = (exp(-(x * x) / s)) / (PI * s);
		sum += kernel[i];
	}

	for (int i = 0; i < KERNEL_SIZE; i++) {
		kernel[i] /= sum;
	}

	cudaMemcpyToSymbol(d_kernel, kernel, KERNEL_SIZE * sizeof(float));
}

int main() {
	unsigned char *h_image = (unsigned char *)malloc(WIDTH * HEIGHT);
	unsigned char *h_blurredImage = (unsigned char *)malloc(WIDTH * HEIGHT);

	// Generate random grayscale image
	for (int i = 0; i < WIDTH * HEIGHT; i++) {
		h_image[i] = rand() % 256;
	}
	stbi_write_png("original.png", WIDTH, HEIGHT, 1, h_image, WIDTH);

	unsigned char *d_image;
	unsigned char *d_blurredImage;
	cudaMalloc(&d_image, WIDTH * HEIGHT);
	cudaMalloc(&d_blurredImage, WIDTH * HEIGHT);

	cudaMemcpy(d_image, h_image, WIDTH * HEIGHT, cudaMemcpyHostToDevice);

	setGaussianKernel(1.0);  // Set sigma for Gaussian Kernel

	dim3 blocks((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	gaussianBlur << < blocks, threads >> > (d_image, d_blurredImage, WIDTH, HEIGHT);

	cudaMemcpy(h_blurredImage, d_blurredImage, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

	stbi_write_png("blurred.png", WIDTH, HEIGHT, 1, h_blurredImage, WIDTH);

	free(h_image);
	free(h_blurredImage);
	cudaFree(d_image);
	cudaFree(d_blurredImage);

	return 0;
}
