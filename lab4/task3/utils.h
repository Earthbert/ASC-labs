#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TRY_CUDA(err) { \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(-1); \
    }} while(0); \


void fill_array_int(int* a, int N) {
    for (int i = 0; i < N; ++i) {
        a[i] = i;
    }
}

void fill_array_float(float* a, int N) {
    for (int i = 0; i < N; ++i) {
        a[i] = (float)i;
    }
}

void fill_array_random(float* a, int N) {
    for (int i = 0; i < N; ++i) {
        a[i] = (float)rand() / RAND_MAX;
    }
}

/**
 * Helper function to check your results.
 *
 * @param task: the number of task you run
 * @param a: pointer to a dynamically allocated array
 */
void check_task_3(int task, int* a) {
    int ok = 1;
    int b[16];

    printf("\n[Checking todo %d]\n", task);

    for (int i = 0; i < 16; ++i) {
        switch (task) {
        case 3:
            b[i] = i % 2;
            break;
        case 4:
            b[i] = i / 4;
            break;
        case 5:
            b[i] = i % 4;
            break;
        }

        ok = ok && (a[i] == b[i]);
    }

    printf("\tYour results: ");
    for (int i = 0; i < 16; ++i) printf("%d ", a[i]);

    printf("\n\tExpected results: ");
    for (int i = 0; i < 16; ++i) printf("%d ", b[i]);

    if (ok)
        printf("\n\tCheck: OK!\n");
    else
        printf("\n\tCheck: WRONG!\n");
}