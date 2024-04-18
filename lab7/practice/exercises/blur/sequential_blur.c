#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define KERNEL_SIZE 5
#define KERNEL_RADIUS (KERNEL_SIZE / 2)
#define WIDTH 512
#define HEIGHT 512

#define PI 3.14159265358979323846

float kernel[KERNEL_SIZE];

void setGaussianKernel(float sigma) {
	float sum = 0.0;
	float s = 2.0 * sigma * sigma;

	for (int i = 0; i < KERNEL_SIZE; i++) {
		int x = i - KERNEL_RADIUS;
		kernel[i] = exp(-(x * x) / s) / (PI * s);
		sum += kernel[i];
	}

	for (int i = 0; i < KERNEL_SIZE; i++) {
		kernel[i] /= sum;
	}
}

void gaussianBlur(const unsigned char *input, unsigned char *output, int width,
	int height) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float sum = 0.0;
			float weightSum = 0.0;
			for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
				for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
					int dX = x + kx;
					int dY = y + ky;
					if (dX >= 0 && dX < width && dY >= 0 && dY < height) {
						float weight = kernel[ky + KERNEL_RADIUS] * kernel[kx + KERNEL_RADIUS];
						sum += input[dY * width + dX] * weight;
						weightSum += weight;
					}
				}
			}
			output[y * width + x] = (unsigned char)(sum / weightSum);
		}
	}
}

int main() {
	unsigned char *h_image = (unsigned char *)malloc(WIDTH * HEIGHT);
	unsigned char *h_blurredImage = (unsigned char *)malloc(WIDTH * HEIGHT);

	// Generate random grayscale image
	for (int i = 0; i < WIDTH * HEIGHT; i++) {
		h_image[i] = rand() % 256;
	}
	stbi_write_png("original.png", WIDTH, HEIGHT, 1, h_image, WIDTH);

	setGaussianKernel(1.0);  // Set sigma for Gaussian Kernel
	gaussianBlur(h_image, h_blurredImage, WIDTH, HEIGHT);
	stbi_write_png("blurred.png", WIDTH, HEIGHT, 1, h_blurredImage, WIDTH);

	free(h_image);
	free(h_blurredImage);

	return 0;
}
