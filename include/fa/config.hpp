#pragma once
#include <cmath>
#include <cstdint>

namespace fa {

constexpr int kWarpSize     = 32;
constexpr int kMaxHeadDim   = 128;   // raise if you need larger D (mind regs/SMEM)
constexpr int kStages       = 2;     // double buffering

struct KernelConfig {
  int head_dim;        // D
  int tile_m;          // query rows per CTA
  int tile_n;          // KV rows per streamed tile
  int block_threads;   // threads per CTA (e.g., 256)
  int loader_warps;    // number of producer warps (1 typical)
  bool causal;         // causal mask
  float scale;         // 1/sqrt(D)
};

struct LaunchConfig {
  // chosen by autotuner or user
  int tile_m   = 224;
  int tile_n   = 64;
  int block    = 256;
  int loaders  = 1;
};

inline KernelConfig make_kernel_cfg(int D, bool causal, const LaunchConfig& lc) {
  KernelConfig cfg{};
  cfg.head_dim      = D;
  cfg.tile_m        = lc.tile_m;
  cfg.tile_n        = lc.tile_n;
  cfg.block_threads = lc.block;
  cfg.loader_warps  = lc.loaders;
  cfg.causal        = causal;
  cfg.scale         = 1.0f / std::sqrt(static_cast<float>(D));
  return cfg;
}

} // namespace fa
