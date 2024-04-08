# Paralelsim dinamic

Idee: să lansăm fire de execuție la rulare, pe măsură ce explorăm spațiul
problemei, care să împartă (Divide et Impera) sarcina fără controlul unității
de comandă.

În această implementare a algoritmului Quicksort, schema de partiționare ne
returnează poziția în vector față de care toate elementele de la stânga sunt
mai mici, respectiv mai mari la dreapta.

Întrucât nu știu unde va fi poziția pivotului, pe care l-am ales mereu să fie
primul element, această problemă este tipică pentru paralelism dinamic.

Implementarea prezentată are mai deficiențe. **NU** ia decizii la rulare în
plus față de un algoritm secvențial și nu ține cont de arhitectura CUDA (ce e
un wrap, cum se partajează memoria, etc.).

Consultați atât un [exemplu simplu], cât și un [exemplu avansat], de
îmbunătățire a implementării curente (provided by CUDA samples).

Am folosit CUDA streams pentru a permite ca sortarea subproblemei stângi să se
execute în paralel cu sortarea subproblemei drepte. Nucleele de execuție copil
sunt executate cu stream-ul implicit (0), care operează blocant față de HOST.
Deci ar fi fost serializate sortările.

[exemplu simplu]: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/cdpSimpleQuicksort/cdpSimpleQuicksort.cu
[exemplu avansat]: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/cdpAdvancedQuicksort/cdpAdvancedQuicksort.cu
