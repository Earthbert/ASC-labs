// TODO
// Inumtirea matricelor

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>     // provides int8_t, uint8_t, int16_t etc.
#include <stdlib.h>
#include <sys/time.h>

#define N 1200
double a[N][N], b[N][N], c[N][N];

int main(int argc, char *argv[]) {

    int bi, bj, bk;
    int i;
    int j;
    int k;
    int blockSize = 50;
    struct timeval start, end;
    float elapsed;

    if (argc > 2) {
        printf("apelati cu %s <n>\n", argv[0]);
        return -1;
    }
    if (argc == 2)
        blockSize = atoi(argv[1]);

    srand(0); //to repeat experiment
    //srand ( time ( NULL)); // if you want random seed


    //reset c matrix and initialize a and b matrix
    for (bi = 0;bi < N;bi++) {
        for (bj = 0;bj < N;bj++) {
            c[bi][bj] = 0.0;
            a[bi][bj] = (double)rand() / RAND_MAX * 2.0 - 1.0; //double in range -1 to 1
            b[bi][bj] = (double)rand() / RAND_MAX * 2.0 - 1.0; //double in range -1 to 1
        }
    }

    gettimeofday(&start, NULL);

    // TODO: implementati i-k-j folosind optimizari

    for (bi = 0; bi < N; bi += blockSize)
        for (bk = 0; bk < N; bk += blockSize)
            for (bj = 0; bj < N; bj += blockSize)
                for (i = bi; i < bi + blockSize; i++)
                    for (k = bk; k < bk + blockSize; k++) {
                        double *c_p = c[i];
                        double *a_p = a[i] + k;
                        double *b_p = b[k] + bj;
                        for (j = bj; j < bj + blockSize; j++) {
                            *c_p += *a_p * *b_p;
                            c_p++;
                            b_p++;
                        }
                    }

    gettimeofday(&end, NULL);

    elapsed = ((end.tv_sec - start.tv_sec) * 1000000.0f + end.tv_usec - start.tv_usec) / 1000000.0f;

    printf("TIME (i-k-j optimize): %12f\n", elapsed);
    return 0;
}
