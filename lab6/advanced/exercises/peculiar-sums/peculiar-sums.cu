#include <cstdlib>
#include <iostream>

#define NUM_ELEM 128
#define BLOCK_SIZE 32

using namespace std;

// TODO
// workers will compute sum on first N elements
__global__ void worker(int *data, int *result, int num) {
	// TODO, compute sum and store in result
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num) {
		int sum = 0;
		for (int i = 0; i < data[idx] && i < num; i++) {
			sum += data[i];
		}
		result[idx] = sum;
	}
}

// TODO
// master will launch threads to compute sum on first N elements
__global__ void master(int *data, int *result, int num) {
	// TODO, schedule worker threads
	worker << <num / BLOCK_SIZE, BLOCK_SIZE >> > (data, result, num);
}

void generateData(int *data, int num) {
	srand(time(0));

	for (int i = 0; i < num; i++) {
		data[i] = rand() % 8 + 2;
	}
}

void print(int *data, int num) {
	for (int i = 0; i < num; i++) {
		cout << data[i] << " ";
	}
	cout << endl;
}

// check
// each element result[i] should be sum of first data[i] elements of data
bool checkResult(int *data, int num, int *result) {
	for (int i = 0; i < num; i++) {
		int sum = 0;
		for (int j = 0; j < data[i] && j < num; j++) {
			sum += data[j];
		}

		if (result[i] != sum) {
			cout << "Error at " << i << ", requested sum of first " << data[i]
				<< " elem, got " << result[i] << endl;
			return false;
		}
	}

	return true;
}

int main(int argc, char *argv[]) {
	int *data = NULL;
	cudaMallocManaged(&data, NUM_ELEM * sizeof(int));

	int *result = NULL;
	cudaMallocManaged(&result, NUM_ELEM * sizeof(int));

	generateData(data, NUM_ELEM);

	// TODO schedule master threads and pass data/result/num
	master << <1, 1 >> > (data, result, NUM_ELEM);
	cudaDeviceSynchronize();

	print(data, NUM_ELEM);
	print(result, NUM_ELEM);

	if (checkResult(data, NUM_ELEM, result)) {
		cout << "Result OK" << endl;
	} else {
		cout << "Result ERR" << endl;
	}

	cudaFree(data);
	cudaFree(result);

	return 0;
}
