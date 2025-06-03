# Block_Matrix_Multiply_CUDA
Block matrix multiply parallel algorithm using CUDA

Compile:
```console
nvcc -O2 bmm_main.cu bmm.cu -o bmm
```

Execute:
```console
./bmm M
```

Note that $N$ is equal to $2^M$.
