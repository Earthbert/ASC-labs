#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void fill_array_int(int *a, int N) {
    for (int i = 0; i < N; ++i) {
        a[i] = i;
    }
}

void fill_array_float(float *a, int N) {
    for (int i = 0; i < N; ++i) {
        a[i] = (float) i;
    }
}

void fill_array_random(float *a, int N) {
    for (int i = 0; i < N; ++i) {
        a[i] = (float) rand() / RAND_MAX;
    }
}

void check_task_4(float *a, float *b, float *c, int N) {
    float eps = 0.001;
    float serial;
    int ok = 1;

    for (int i = 0; i < N; ++i) {
        serial = a[i] + b[i];
        if (abs(serial - c[i]) > eps) {
            printf("WRONG value at index %d: [value: %.4f] [expected: %.4f]!\n",
                i, c[i], serial);
            ok = 0;
        }
    }

    printf("\na[i]\tb[i]\tc[i]\texpected\n");
    for (int i = 0; i < 10; ++i) {
        printf("%1.3f\t%1.3f\t%1.3f\t%1.3f\n", a[i], b[i], c[i], a[i] + b[i]);
    }

    if (ok)
        printf("\nCheck: OK!\n");
    else
        printf("\nCheck: WRONG!\n");
}