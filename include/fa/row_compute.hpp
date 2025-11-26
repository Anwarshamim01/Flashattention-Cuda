#pragma once
#include "traits.hpp"
#include "softmax.hpp"
#include "config.hpp"

namespace fa {

__device__ __forceinline__ bool mask_ok(bool causal, int kj, int qi) {
  return !(causal && kj > qi);
}

template<typename T>
struct RowComputer {
  using Storage = typename TypeTraits<T>::Storage;

  static __device__ void consume_tile_for_row(
      const float* __restrict__ q_reg,  // [D]
      int D, int qi_global, int k_row_start, int tile_n,
      bool causal, float scale,
      const Storage* __restrict__ K_tile, // [tile_n, D] in Storage=T
      const Storage* __restrict__ V_tile, // [tile_n, D] in Storage=T
      SoftmaxRowState& st, float* __restrict__ acc)
  {
    // Pass 1: tile max
    float max_tile = -INFINITY;
    for (int j=0; j<tile_n; ++j) {
      const int kj = k_row_start + j;
      if (!mask_ok(causal, kj, qi_global)) continue;
      float dot = 0.f;
      const Storage* Kj = &K_tile[j*D];
      #pragma unroll
      for (int c=0; c<kMaxHeadDim; ++c) {
        if (c>=D) break;
        dot += q_reg[c] * TypeTraits<T>::to_float(Kj[c]);
      }
      dot *= scale;
      if (dot > max_tile) max_tile = dot;
    }
    float m_new = fmaxf(st.m, max_tile);
    if (!isfinite(m_new)) return; // fully masked
    st.rescale(m_new, acc, D);

    // Pass 2: sums
    for (int j=0; j<tile_n; ++j) {
      const int kj = k_row_start + j;
      if (!mask_ok(causal, kj, qi_global)) continue;

      const Storage* Kj = &K_tile[j*D];
      float dot = 0.f;
      #pragma unroll
      for (int c=0; c<kMaxHeadDim; ++c) {
        if (c>=D) break;
        dot += q_reg[c] * TypeTraits<T>::to_float(Kj[c]);
      }
      float s = fastexp(dot * scale - st.m);
      st.l += s;

      const Storage* Vj = &V_tile[j*D];
      #pragma unroll
      for (int c=0; c<kMaxHeadDim; ++c) {
        if (c>=D) break;
        acc[c] += s * TypeTraits<T>::to_float(Vj[c]);
      }
    }
  }
};

// (Optional) WMMA fragments path: you can add a specialized consumer that forms a
// 16x16 logits tile per warp and updates 16 rows at once. Kept as a hook so the
// default code stays portable & simple.
#ifdef FA_USE_WMMA
// See forward_kernels.cu for an example hook.
#endif

} // namespace fa
