# Timing

## Via execuție kernel (host/CPU)

Putem masură timpul de execuție al diverselor operații (execuție kernel,
transfer date, etc.), când acestea sunt *blocante*. Astfel obținem timpi de
execuție al operațiilor din perspectiva unității host (CPU). Această metodă
nu este foarte precisă, deoarece în timpul de execuție sunt incluse și toate
operațiile de control CPU ↔ GPU.

În [cpu_timing.cu](cpu_timing.cu) avem un exemplu de folosire a functiei
`cudaDeviceSynchronize` pentru a forta o blocare pe partea de host/CPU până
când toate operațiile de pe GPU au fost executate. 

## Via CUDA events (device/GPU)

O variantă mai exactă decât operațiile blocante sunt CUDA events. Acestea
au suport hardware și oferă timpi de execuție din perspectiva unității device 
(GPU). Mai jos avem un exemplu folosind CUDA events. 

Urmăriți exemplul din [events_timing.cu](events_timing.cu) în acest sens.
