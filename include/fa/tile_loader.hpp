#pragma once
#include <stdint.h>
#include "tensor.hpp"
#include "traits.hpp"
#include "config.hpp"

namespace fa {

__device__ __forceinline__ int lane_id() { return threadIdx.x & (kWarpSize-1); }
__device__ __forceinline__ int warp_id() { return threadIdx.x >> 5; }

// Generic 16B vector copy for aligned pointers (any type)
__device__ __forceinline__ void memcpy_vec16(void* __restrict__ dst,
                                             const void* __restrict__ src,
                                             size_t bytes)
{
  const int lane = lane_id();
  const size_t n = bytes / 16;
  const size_t tail = bytes % 16;
  auto* __restrict__ d = reinterpret_cast<uint4*>(dst);
  const auto* __restrict__ s = reinterpret_cast<const uint4*>(src);
  for (size_t i = lane; i < n; i += kWarpSize) d[i] = s[i];
  if (lane == 0 && tail) {
    // tail copy
    const uint8_t* sb = reinterpret_cast<const uint8_t*>(src) + n*16;
    uint8_t*       db = reinterpret_cast<uint8_t*>(dst) + n*16;
    for (size_t t = 0; t < tail; ++t) db[t] = sb[t];
  }
}

// Optional cp.async helpers (SM80+)
#if defined(FA_USE_CP_ASYNC) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr){
  unsigned smem_u32 = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit(){ asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait(){   asm volatile("cp.async.wait_group 0;\n"); }
#endif

template<typename T>
struct TileLoader {
  using Storage = typename TypeTraits<T>::Storage;

  // Load a contiguous [tile_n x D] slice starting at (b,h,k0,0)
  // into shared buffers (typed as Storage = T, not float).
  static __device__ void load_tile(
      const Tensor4D<T>& K, const Tensor4D<T>& V,
      int b, int h, int k0, int tile_n, int D,
      Storage* __restrict__ K_buf, Storage* __restrict__ V_buf,
      int loader_warps)
  {
    const int wid = warp_id();
    const int lane = lane_id();
    if (wid >= loader_warps) return;  // only producer warps run this

    const size_t bytes = static_cast<size_t>(tile_n) * D * sizeof(Storage);
    const Storage* gK  = reinterpret_cast<const Storage*>(K.row_ptr(b,h,k0));
    const Storage* gV  = reinterpret_cast<const Storage*>(V.row_ptr(b,h,k0));

#if defined(FA_USE_CP_ASYNC) && (__CUDA_ARCH__ >= 800)
    // cp.async path (warp-level)
    {
      const uint8_t* srcK = reinterpret_cast<const uint8_t*>(gK);
      const uint8_t* srcV = reinterpret_cast<const uint8_t*>(gV);
      uint8_t* dstK = reinterpret_cast<uint8_t*>(K_buf);
      uint8_t* dstV = reinterpret_cast<uint8_t*>(V_buf);

      const size_t n = bytes / 16;
      const size_t tail = bytes % 16;

      for (size_t i = lane; i < n; i += kWarpSize) {
        cp_async_16(dstK + i*16, srcK + i*16);
        cp_async_16(dstV + i*16, srcV + i*16);
      }
      cp_async_commit();
      cp_async_wait(); // needed before consumers read this buffer
      if (lane == 0 && tail) {
        for (size_t t=0; t<tail; ++t) {
          dstK[n*16 + t] = srcK[n*16 + t];
          dstV[n*16 + t] = srcV[n*16 + t];
        }
      }
    }
#else
    // Vectorized memcpy when 16B-aligned and size multiple of 16; else scalar fallback.
    uintptr_t aK = reinterpret_cast<uintptr_t>(gK);
    uintptr_t aV = reinterpret_cast<uintptr_t>(gV);
    uintptr_t aKs = reinterpret_cast<uintptr_t>(K_buf);
    uintptr_t aVs = reinterpret_cast<uintptr_t>(V_buf);
    const bool vec_ok = ((aK|aV|aKs|aVs) % 16 == 0) && (bytes % 16 == 0);
    if (vec_ok) {
      memcpy_vec16(K_buf, gK, bytes);
      memcpy_vec16(V_buf, gV, bytes);
    } else {
      // scalar strided copy
      const int elems = tile_n * D;
      for (int idx = lane; idx < elems; idx += kWarpSize) {
        K_buf[idx] = gK[idx];
        V_buf[idx] = gV[idx];
      }
    }
#endif
  }
};

} // namespace fa
