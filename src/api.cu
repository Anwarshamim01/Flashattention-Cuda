#include "fa/api.hpp"
#include "fa/forward_kernels.cuh"
#include "fa/config.hpp"
#include "fa/tensor.hpp"
#include "fa/traits.hpp"
#include <cuda_runtime.h>

namespace fa {

template<typename T>
void flash_attention_forward(
    T* q, T* k, T* v, T* o,
    int B, int H, int S, int D,
    bool causal,
    const LaunchConfig& lcfg,
    cudaStream_t stream)
{
  Tensor4D<T> Q(q,B,H,S,D), K(k,B,H,S,D), V(v,B,H,S,D), O(o,B,H,S,D);
  KernelConfig cfg = make_kernel_cfg(D, causal, lcfg);

  dim3 grid( (S + cfg.tile_m - 1)/cfg.tile_m, H, B );
  const size_t smem = static_cast<size_t>(cfg.tile_n) * cfg.head_dim * 4 * sizeof(typename TypeTraits<T>::Storage);

  // opt-in to large dynamic smem (safe no-op on some GPUs)
  cudaFuncSetAttribute(forward_row_split_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaFuncSetAttribute(forward_row_split_kernel<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  forward_row_split_kernel<T><<<grid, cfg.block_threads, smem, stream>>>(Q,K,V,O,cfg);
}

template void flash_attention_forward<float>(float*,float*,float*,float*,int,int,int,int,bool,const LaunchConfig&,cudaStream_t);
template void flash_attention_forward<__half>(__half*,__half*,__half*,__half*,int,int,int,int,bool,const LaunchConfig&,cudaStream_t);
#ifdef __CUDA_BF16_TYPES_EXIST__
template void flash_attention_forward<__nv_bfloat16>(__nv_bfloat16*,__nv_bfloat16*,__nv_bfloat16*,__nv_bfloat16*,int,int,int,int,bool,const LaunchConfig&,cudaStream_t);
#endif

template<typename T>
void flash_attention_forward_splitk(
    T* q, T* k, T* v, T* o,
    int B, int H, int S, int D,
    bool causal,
    int k_splits,
    void* workspace, size_t workspace_bytes,
    const LaunchConfig& lcfg,
    cudaStream_t stream)
{
  Tensor4D<T> Q(q,B,H,S,D), K(k,B,H,S,D), V(v,B,H,S,D), O(o,B,H,S,D);
  KernelConfig cfg = make_kernel_cfg(D, causal, lcfg);

  // workspace layout: [splits, B, H, S] for m, l, and acc[D]
  size_t rows_per_split = static_cast<size_t>(B)*H*S;
  size_t m_bytes = rows_per_split * sizeof(float);
  size_t l_bytes = rows_per_split * sizeof(float);
  
  
  // size_t acc_bytes = rows_per_split * D * sizeof(float);

  uint8_t* base = reinterpret_cast<uint8_t*>(workspace);
  float* m_ptr0 = reinterpret_cast<float*>(base);
  float* l_ptr0 = reinterpret_cast<float*>(base + k_splits*m_bytes);
  float* acc_ptr0 = reinterpret_cast<float*>(base + k_splits*m_bytes + k_splits*l_bytes);

  const size_t smem = static_cast<size_t>(cfg.tile_n) * cfg.head_dim * 4 * sizeof(typename TypeTraits<T>::Storage);

  // Launch partials (one after another for clarity; you can also multi‑stream)
  int keys_per_split = (S + k_splits - 1)/k_splits;
  for (int s=0; s<k_splits; ++s) {
    int k_begin = s * keys_per_split;
    int k_end   = min(S, (s+1)*keys_per_split);
    if (k_begin >= k_end) continue;

    float* m_s   = m_ptr0   + s * rows_per_split;
    float* l_s   = l_ptr0   + s * rows_per_split;
    float* acc_s = acc_ptr0 + s * rows_per_split * D;

    dim3 grid( (S + cfg.tile_m - 1)/cfg.tile_m, H, B );
    cudaFuncSetAttribute(forward_splitk_partial_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaFuncSetAttribute(forward_splitk_partial_kernel<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    forward_splitk_partial_kernel<T><<<grid, cfg.block_threads, smem, stream>>>(
        Q,K,V, m_s, l_s, acc_s, k_begin, k_end, cfg);
  }

  // Merge kernel: 1D over S (rows)
  dim3 grid_merge( (S + 127)/128, H, B );
  dim3 block_merge(128);
  forward_splitk_merge_kernel<T><<<grid_merge, block_merge, 0, stream>>>(
      m_ptr0, l_ptr0, acc_ptr0, O, k_splits, cfg);
}

template void flash_attention_forward_splitk<float>(float*,float*,float*,float*,int,int,int,int,bool,int,void*,size_t,const LaunchConfig&,cudaStream_t);
template void flash_attention_forward_splitk<__half>(__half*,__half*,__half*,__half*,int,int,int,int,bool,int,void*,size_t,const LaunchConfig&,cudaStream_t);
#ifdef __CUDA_BF16_TYPES_EXIST__
template void flash_attention_forward_splitk<__nv_bfloat16>(__nv_bfloat16*,__nv_bfloat16*,__nv_bfloat16*,__nv_bfloat16*,int,int,int,int,bool,int,void*,size_t,const LaunchConfig&,cudaStream_t);
#endif

} // namespace fa