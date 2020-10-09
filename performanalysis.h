

#ifndef PERFORMANALYSIS_H_
#define PERFORMANALYSIS_H_

#include<stdio.h>
#include<cuda.h>
#include<math.h>
#include<cuda_runtime.h>
#include<stdlib.h>

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define cublasSafeCall(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}


#define checkCusparse(call)                                                    \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}


__global__ void initialize(double*, double*, double*, double*, double*, double*, int, int,  int);

__global__ void pcr_penta(double*, double*, double*, double*, double*, double*, int, int);

__global__ void shared_cached_shared_tile_splitting(double*, double*, double*, double*, double*, double*, int, int, int, int, int, int);
__global__ void global_cached_shared_tile_splitting(double*, double*, double*, double*, double*, double*, int, int, int, int);
__global__ void global_cached_global_tile_splitting(double*, double*, double*, double*, double*, double*, int, int, int, int);

__global__ void pcr_penta_splitting_2nd_stage(double*, double*, double*, double*, double*, double*, int, int, int, int);


#endif
