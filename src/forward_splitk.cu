#include "fa/forward_kernels.cuh"
#include "fa/traits.hpp"
#include "fa/tensor.hpp"
#include "fa/config.hpp"
#include "fa/softmax.hpp"
#include "fa/row_compute.hpp"
#include "fa/tile_loader.hpp"

namespace fa {

// Partial: same as row-split kernel, but restricted to [k_begin, k_end) keys
// and writing (m,l,acc) per row to workspace instead of final O.
template<typename T>
__global__ void forward_splitk_partial_kernel(
    Tensor4D<T> Q, Tensor4D<T> K, Tensor4D<T> V,
    float* m_buf, float* l_buf, float* acc_buf,
    int k_begin, int k_end, KernelConfig cfg)
{
  using Storage = typename TypeTraits<T>::Storage;
  extern __shared__ uint8_t smem_raw[];
  Storage* K_s0 = reinterpret_cast<Storage*>(smem_raw);
  Storage* K_s1 = K_s0 + cfg.tile_n * cfg.head_dim;
  Storage* V_s0 = K_s1 + cfg.tile_n * cfg.head_dim;
  Storage* V_s1 = V_s0 + cfg.tile_n * cfg.head_dim;

  const int b = blockIdx.z, h = blockIdx.y;
  const int q_block_row = blockIdx.x * cfg.tile_m;

  const int wid = threadIdx.x / kWarpSize;
  const bool is_loader = (wid < cfg.loader_warps);
  const int compute_base = cfg.loader_warps * kWarpSize;
  const int mi = threadIdx.x - compute_base;              // row within tile_m
  const int qi = q_block_row + mi;
  const bool row_active = (threadIdx.x >= compute_base) && (mi < cfg.tile_m) && (qi < Q.S());

  float q_reg[kMaxHeadDim]; SoftmaxRowState st; float acc[kMaxHeadDim];
  if (row_active) {
    #pragma unroll
    for (int c=0; c<kMaxHeadDim; ++c) if (c<cfg.head_dim) q_reg[c] = TypeTraits<T>::to_float(Q(b,h,qi,c));
    st.reset();
    #pragma unroll
    for (int c=0; c<kMaxHeadDim; ++c) if (c<cfg.head_dim) acc[c] = 0.f;
  }

  // prime
  if (is_loader) {
    const int k0 = k_begin;
    const int tn = min(cfg.tile_n, k_end - k0);
    TileLoader<T>::load_tile(K, V, b, h, k0, tn, cfg.head_dim, K_s0, V_s0, cfg.loader_warps);
  }
  __syncthreads();

  int read_buf = 0, write_buf = 1;

  for (int k0 = k_begin; k0 < k_end; k0 += cfg.tile_n) {
    const int tn_read = min(cfg.tile_n, k_end - k0);

    if (row_active) {
      const Storage* K_rd = (read_buf==0) ? K_s0 : K_s1;
      const Storage* V_rd = (read_buf==0) ? V_s0 : V_s1;

      RowComputer<T>::consume_tile_for_row(
          q_reg, cfg.head_dim, qi, k0, tn_read, cfg.causal, cfg.scale,
          K_rd, V_rd, st, acc);
    }

    const bool has_next = (k0 + cfg.tile_n < k_end);
    if (is_loader && has_next) {
      const int next_k0 = k0 + cfg.tile_n;
      const int tn = min(cfg.tile_n, k_end - next_k0);
      Storage* K_wr = (write_buf==0) ? K_s0 : K_s1;
      Storage* V_wr = (write_buf==0) ? V_s0 : V_s1;
      TileLoader<T>::load_tile(K, V, b, h, next_k0, tn, cfg.head_dim, K_wr, V_wr, cfg.loader_warps);
    }

    __syncthreads();
    int tmp = read_buf; read_buf = write_buf; write_buf = tmp;
  }

  // write partials: layout [split, B, H, S]
  // Instead: the host passes separate base pointers for each split range, so
  // we write to offset 0 here. The host offsets the pointers per-split.
  if (row_active) {
    // m, l, acc[D]
    size_t base = ((size_t)b * Q.H() + h) * Q.S() + qi;
    m_buf[base] = st.m;  l_buf[base] = st.l;
    float* acc_row = acc_buf + base * cfg.head_dim;
    #pragma unroll
    for (int c=0; c<kMaxHeadDim; ++c) if (c<cfg.head_dim) acc_row[c] = acc[c];
  }
}

template<typename T>
__global__ void forward_splitk_merge_kernel(
    float* m_buf, float* l_buf, float* acc_buf,
    Tensor4D<T> O, int splits, KernelConfig cfg)
{
  const int b = blockIdx.z, h = blockIdx.y;
  const int qi = blockIdx.x * blockDim.x + threadIdx.x;
  if (qi >= O.S()) return;

  // merge over splits for (b,h,qi)
  const size_t row_base = ((size_t)b * O.H() + h) * O.S() + qi;
  float m = -INFINITY;
  // 1) max of m
  for (int s=0; s<splits; ++s) {
    const size_t off = row_base + (size_t)s * (O.B()*O.H()*O.S()); // host arranges split stride
    m = fmaxf(m, m_buf[off]);
  }
  float l = 0.f;
  // temp accumulator in registers
  float acc[kMaxHeadDim]; for (int c=0;c<kMaxHeadDim;++c) if (c<cfg.head_dim) acc[c]=0.f;

  for (int s=0; s<splits; ++s) {
    const size_t off = row_base + (size_t)s * (O.B()*O.H()*O.S());
    float ms = m_buf[off];
    float ls = l_buf[off];
    float alpha = (isfinite(ms)) ? __expf(ms - m) : 0.f;
    l += alpha * ls;
    const float* acc_s = acc_buf + (off * cfg.head_dim);
    #pragma unroll
    for (int c=0;c<kMaxHeadDim;++c) if (c<cfg.head_dim) acc[c] += alpha * acc_s[c];
  }

  float inv_l = (l>0.f) ? 1.f / l : 0.f;
  for (int c=0;c<kMaxHeadDim;++c) if (c<cfg.head_dim) {
    O(b,h,qi,c) = TypeTraits<T>::from_float(acc[c] * inv_l);
  }
}

#define INSTANTIATE_SPLITK(T) \
  template __global__ void forward_splitk_partial_kernel<T>(Tensor4D<T>,Tensor4D<T>,Tensor4D<T>,float*,float*,float*,int,int,KernelConfig); \
  template __global__ void forward_splitk_merge_kernel<T>(float*,float*,float*,Tensor4D<T>,int,KernelConfig);

INSTANTIATE_SPLITK(float)
INSTANTIATE_SPLITK(__half)
#ifdef __CUDA_BF16_TYPES_EXIST__
INSTANTIATE_SPLITK(__nv_bfloat16)
#endif

} // namespace fa