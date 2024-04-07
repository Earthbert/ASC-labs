# Debug

Cele mai des întâlnite probleme sunt cele de acces invalid la memorie. Nu de
puține ori vom observa ca aceste accesuri invalide pot crea efecte secundare
(sau laterale) sau erori ce apar/sunt semnalate abia ulterior.

## Puține accesuri invalide

Să luăm de exemplu cazul în care un thread accesează un element de date. Dăm în
execuție mai multe thread-uri decât elemente de memorie alocate, în primul caz.

```C
#define MAGNITUDE   1
```

Am lansat $8 \times 16 = 128$ thread-uri ca să acceseze $100$ elemente.

Dacă rulăm programul vom observa că nu întoarce nici o eroare. Deoarece sunt
puține accesuri invalide.

CUDA însă oferă aplicații care să analizeze și să detecteze accesuri invalide
cu o precizie ridicată. Dacă rulăm, de exemplu, `compute-sanitizer` vom vedea
instant că avem accesuri invalide la memorie. 

```bash
[prenume.numeID@fep8 debug]$ make
nvcc   debug.cu -o debug
[prenume.numeID@fep8 debug]$ make run
Submitted batch job 341649
[prenume.numeID@fep8 debug]$ cat slurm-341649.out
# empty string
[prenume.numeID@fep8 ~]$ make run RUN_CMD='compute-sanitizer ./$${exec}'
Submitted batch job 341651
[prenume.numeID@fep8 debug]$ cat slurm-341651.out
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at 0x78 in kernel_compute(int *)
=========     by thread (128,0,0) in block (0,0,0)
=========     Address 0x7f9869a00200 is out of bounds
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame: [0x2ef370]
=========                in /lib64/libcuda.so.1
=========     Host Frame: [0x875b]
=========                in /export/home/acs/stud/m/prenume.numeID/asc-public/labs/cuda/arch/tutorials/debug/./debug
...
unspecified launch failure in debug.cu at line 33
========= Target application returned an error
========= ERROR SUMMARY: 1921 errors
```

## Multe accesuri invalide

Setați `MAGNITUDE` la o valoare mai mare.

```C
#define MAGNITUDE   (1024 * 1024)
```

O să observați că acum la rulare se afișează o eroare, acum. Folosim `cuda-gdb`
ca să investigăm problema.

```bash
[prenume.numeID@fep8 debug]$ make run
Submitted batch job 341672
[prenume.numeID@fep8 debug]$ cat slurm-341672.out
an illegal memory access was encountered in debug.cu at line 33
[prenume.numeID@fep8 ~]$ make run RUN_CMD='cuda-gdb -x gdb_commands.txt ./$${exec}'
Submitted batch job 341673
[prenume.numeID@fep8 debug]$ cat slurm-341673.out
NVIDIA (R) CUDA Debugger
...
CUDA Exception: Warp Illegal Address

Thread 1 "debug" received signal CUDA_EXCEPTION_14, Warp Illegal Address.
[Switching focus to CUDA kernel 0, grid 1, block (418462,0,0), thread (128,0,0), device 0, sm 5, warp 3, lane 0]
0x00000100002f0a08 in kernel_compute(int*)<<<(8388608,1,1),(256,1,1)>>> ()
```
