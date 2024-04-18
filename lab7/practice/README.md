# Practică

## Recapitulare

Înainte să vă apucați de acest laborator vă rugăm să verificați că sunteți
familiari cu dezvoltarea folosind cluster-ul/unitatea personală pe baza
surselor din acest repo.

Recitiți [descrierea infrastructurii](../README.md) în acest sens, în special
partea despre [profiling].

## Exerciții

0. Rulați exemplul [quicksort] din laboratorul trecut.
1. Rezolvați problemele din [exercises](exercises/) urmărind `TODO`-urile.
    1. Implementați algoritmul [OETS].
    1. Implementați algoritmul [merge sort].
        > Opțional. Studiați implementarea din [CUDA Samples mergeSort] și
        > însușiți-vă tehnicile de optimizare.
    1. Finalizați cu [Gaussian Blur] "traducând" implementarea [secvențială].
        > Observație. `make` va compila *mai întâi* implementarea [secvențială]
        > și varianta în CUDA, care va genera erori de compilare dacă nu este
        > completă.
        >
        > Utilizați `make run EXEC='sequential_blur'` ca să rulați implementarea
        > [secvențială] și `make run` simplu, pentru a doua.
        >
        > Utilizați `make clean-all` pentru curățarea acestui exercițiu.

        > Opțional. Studiați impelementarea din [CUDA Samples recursiveGaussian]
        > și însușiți-vă tehnicile de optimizare.


[profiling]: ../README.md#debugging-memchecking-and-profiling

[quicksort]: ../advanced/tutorials/quicksort/

[OETS]: exercises/oets
[merge sort]: exercises/mergesort

[CUDA Samples mergeSort]: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/mergeSort

[Gaussian Blur]: exercises/blur
[secvențială]: exercises/blur/sequential_blur.c

[CUDA Samples recursiveGaussian]: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/recursiveGaussian
