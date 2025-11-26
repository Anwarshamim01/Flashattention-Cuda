#pragma once
#include <cuda_fp16.h>
#ifdef __CUDACC__
  #include <cuda_bf16.h>
#endif

namespace fa {

template<typename T> struct TypeTraits {};

template<> struct TypeTraits<float> {
  using Storage = float;
  static __device__ __forceinline__ float to_float(float x){ return x; }
  static __device__ __forceinline__ float from_float(float x){ return x; }
};

template<> struct TypeTraits<__half> {
  using Storage = __half;
  static __device__ __forceinline__ float   to_float(__half x){ return __half2float(x); }
  static __device__ __forceinline__ __half from_float(float x){ return __float2half(x); }
};

#ifdef __CUDA_BF16_TYPES_EXIST__
template<> struct TypeTraits<__nv_bfloat16> {
  using Storage = __nv_bfloat16;
  static __device__ __forceinline__ float          to_float(__nv_bfloat16 x){ return __bfloat162float(x); }
  static __device__ __forceinline__ __nv_bfloat16 from_float(float x){ return __float2bfloat16(x); }
};
#endif

__device__ __forceinline__ float fastexp(float x) { return __expf(x); }

} // namespace fa
