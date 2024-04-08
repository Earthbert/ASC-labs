# Programare avansată CUDA

## Recapitulare

Înainte să vă apucați de acest laborator vă rugăm să verificați că sunteți
familiari cu dezvoltarea folosind cluster-ul/unitatea personală a surselor
din acest repo.

Recitiți [descrierea infrastructurii](../README.md) în acest sens, în special
partea despre [profiling].

## Exerciții

0. Rulați exemplele din [tutorials](tutorials/).
   > *Notă*. Analizați comenzile `sbatch` rulate de `Makefile` pentru a
    înțelege care este modul de operare/execuție în cluster.
1. Rezolvați problemele din [exercises](exercises/) urmărind `TODO`-urile.
    1. Implementați o operație [trigo]nmetrică în paradigma [no-unified], cât și
      [unified]. Analizați performanța diferitelor implementări prin [profiling].
    2. Calculați niște statistici ([stats]) simple pe un vector, în trei
      paradigme diferite: [no-atomics], [partial-atomics] și [full-atomics].
      Analizați performanța diferitelor implementări prin [profiling].
    3. Continuați antrenamentul cu [inserts].
    4. Finalizați cu [peculiar-sums].

[profiling]: ../README.md#debugging-memchecking-and-profiling

[trigo]: exercises/trigo/
[no-unified]: exercises/trigo/no-unified
[unified]: exercises/trigo/unified

[stats]: exercises/stats
[no-atomics]: exercises/stats/no-atomics
[partial-atomics]: exercises/stats/partial-atomics
[full-atomics]: exercises/stats/full-atomics

[inserts]: exercises/inserts

[peculiar-sums]: exercises/peculiar-sums
