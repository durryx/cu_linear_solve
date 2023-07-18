#include <stdio.h>


#define CHECK_KERNELCALL()                                                     \
    {                                                                          \
        const cudaError_t err = cudaGetLastError();                            \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, \
                   __LINE__);                                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }


__device__ char my_char_atomic_flag(char *addr){
  unsigned *al_addr = reinterpret_cast<unsigned *> (((unsigned long long)addr) & (0xFFFFFFFFFFFFFFFCULL));
  unsigned al_offset = ((unsigned)(((unsigned long long)addr) & 3)) * 8;
  unsigned my_bit = 1U << al_offset;
  return (char) ((atomicOr(al_addr, my_bit) >> al_offset) & 0xFFU);
}

__global__ void k(){

  __shared__ char flag[1024];
  flag[threadIdx.x] = 0;
  __syncthreads();

  int retval = my_char_atomic_flag(flag+(threadIdx.x>>1));
  printf("thread %d saw flag as %d\n", threadIdx.x, retval);
}


int main(){
  k<<<1,1024>>>();
  CHECK_KERNELCALL();    
  cudaDeviceSynchronize();
}
