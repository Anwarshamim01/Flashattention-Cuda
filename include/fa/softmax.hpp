#pragma once
#include "traits.hpp"
#include "config.hpp"

namespace fa {

struct SoftmaxRowState {
  float m;  // running max
  float l;  // running denominator
  __device__ __forceinline__ void reset(){ m = -INFINITY; l = 0.f; }
  __device__ __forceinline__ void rescale(float m_new, float* acc, int D){
    float alpha = fastexp(m - m_new);
    l *= alpha;
    #pragma unroll
    for (int c = 0; c < kMaxHeadDim; ++c) { if (c<D) acc[c] *= alpha; }
    m = m_new;
  }
};

} // namespace fa
