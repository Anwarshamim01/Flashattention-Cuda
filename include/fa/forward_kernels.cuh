#pragma once
#include "tensor.hpp"
#include "config.hpp"

namespace fa {

// Row-split (multi-CTA per head) kernel
template<typename T>
__global__ void forward_row_split_kernel(
    Tensor4D<T> Q, Tensor4D<T> K, Tensor4D<T> V, Tensor4D<T> O,
    KernelConfig cfg);

// Split-K (advanced): partial compute over key ranges, write (m,l,acc) to workspace
template<typename T>
__global__ void forward_splitk_partial_kernel(
    Tensor4D<T> Q, Tensor4D<T> K, Tensor4D<T> V,
    float* m_buf, float* l_buf, float* acc_buf, // [splits,B,H,S,(1 or D)]
    int k_begin, int k_end, KernelConfig cfg);

// Merge partials across splits and write final O
template<typename T>
__global__ void forward_splitk_merge_kernel(
    float* m_buf, float* l_buf, float* acc_buf,
    Tensor4D<T> O,
    int splits, KernelConfig cfg);

} // namespace fa
