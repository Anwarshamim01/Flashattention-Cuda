#include "fa/forward_kernels.cuh"
#include "fa/traits.hpp"
#include "fa/tensor.hpp"
#include "fa/config.hpp"
#include "fa/softmax.hpp"
#include "fa/tile_loader.hpp"
#include "fa/row_compute.hpp"

namespace fa {

template<typename T>
__global__ void forward_row_split_kernel(
    Tensor4D<T> Q, Tensor4D<T> K, Tensor4D<T> V, Tensor4D<T> O,
    KernelConfig cfg)
{
  using Storage = typename TypeTraits<T>::Storage;

  extern __shared__ uint8_t smem_raw[];
  // Shared layout: [K0 | K1 | V0 | V1] typed as Storage
  Storage* K_s0 = reinterpret_cast<Storage*>(smem_raw);
  Storage* K_s1 = K_s0 + cfg.tile_n * cfg.head_dim;
  Storage* V_s0 = K_s1 + cfg.tile_n * cfg.head_dim;
  Storage* V_s1 = V_s0 + cfg.tile_n * cfg.head_dim;

  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int q_block_row = blockIdx.x * cfg.tile_m;

  const int wid = threadIdx.x / kWarpSize;
  const int tid_in_block = threadIdx.x;
  const bool is_loader = (wid < cfg.loader_warps);

  // Compute threads are the rest of the block
  const int compute_base = cfg.loader_warps * kWarpSize;
  const int compute_tid  = tid_in_block - compute_base;
  const bool is_compute  = (tid_in_block >= compute_base);

  // Each compute thread owns at most one row in this CTA
  const int mi = compute_tid;                     // local row id
  const int qi = q_block_row + mi;                // global row id
  const bool row_active = is_compute && (mi < cfg.tile_m) && (qi < Q.S());

  // Per-thread row state
  float q_reg[kMaxHeadDim];
  SoftmaxRowState st; float acc[kMaxHeadDim];
  if (row_active) {
    #pragma unroll
    for (int c=0; c<kMaxHeadDim; ++c) if (c < cfg.head_dim)
      q_reg[c] = TypeTraits<T>::to_float(Q(b,h,qi,c));
    st.reset();
    #pragma unroll
    for (int c=0; c<kMaxHeadDim; ++c) if (c < cfg.head_dim) acc[c] = 0.f;
  }

  // Prime stage 0
  if (is_loader) {
    const int k0 = 0;
    const int tn = min(cfg.tile_n, K.S() - k0);
    TileLoader<T>::load_tile(K, V, b, h, k0, tn, cfg.head_dim, K_s0, V_s0, cfg.loader_warps);
  }
  __syncthreads();

  int read_buf = 0, write_buf = 1;

  for (int k0 = 0; k0 < K.S(); k0 += cfg.tile_n) {
    const int tile_n_read = min(cfg.tile_n, K.S() - k0);

    // Consumers compute on read buffer
    if (row_active) {
      const Storage* K_rd = (read_buf==0) ? K_s0 : K_s1;
      const Storage* V_rd = (read_buf==0) ? V_s0 : V_s1;

      RowComputer<T>::consume_tile_for_row(
          q_reg, cfg.head_dim, qi, k0, tile_n_read, cfg.causal, cfg.scale,
          K_rd, V_rd, st, acc);
    }

    // Producers prefetch next tile into write buffer
    const bool has_next = (k0 + cfg.tile_n < K.S());
    if (is_loader && has_next) {
      const int next_k0 = k0 + cfg.tile_n;
      const int tn = min(cfg.tile_n, K.S() - next_k0);
      Storage* K_wr = (write_buf==0) ? K_s0 : K_s1;
      Storage* V_wr = (write_buf==0) ? V_s0 : V_s1;
      TileLoader<T>::load_tile(K, V, b, h, next_k0, tn, cfg.head_dim, K_wr, V_wr, cfg.loader_warps);
    }

    __syncthreads();
    int tmp = read_buf; read_buf = write_buf; write_buf = tmp;
  }

  // Write out
  if (row_active) {
    float inv_l = (st.l > 0.f) ? 1.f / st.l : 0.f;
    #pragma unroll
    for (int c=0; c<kMaxHeadDim; ++c) if (c < cfg.head_dim) {
      float outc = acc[c] * inv_l;
      O(b,h,qi,c) = TypeTraits<T>::from_float(outc);
    }
  }
}

#define INSTANTIATE_ROW_SPLIT(T) \
  template __global__ void forward_row_split_kernel<T>(Tensor4D<T>,Tensor4D<T>,Tensor4D<T>,Tensor4D<T>,KernelConfig);

INSTANTIATE_ROW_SPLIT(float)
INSTANTIATE_ROW_SPLIT(__half)
#ifdef __CUDA_BF16_TYPES_EXIST__
INSTANTIATE_ROW_SPLIT(__nv_bfloat16)
#endif

} // namespace fa
