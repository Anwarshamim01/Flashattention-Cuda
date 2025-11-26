#pragma once
#include "tensor.hpp"
#include "config.hpp"
#include "workspace.hpp"

namespace fa {

template<typename T>
void flash_attention_forward(
    T* q, T* k, T* v, T* o,
    int B, int H, int S, int D,
    bool causal,
    const LaunchConfig& lcfg,
    cudaStream_t stream = 0);

// Split-K variant (k_splits > 1)
template<typename T>
void flash_attention_forward_splitk(
    T* q, T* k, T* v, T* o,
    int B, int H, int S, int D,
    bool causal,
    int k_splits,
    void* workspace, size_t workspace_bytes,
    const LaunchConfig& lcfg,
    cudaStream_t stream = 0);

} // namespace fa
