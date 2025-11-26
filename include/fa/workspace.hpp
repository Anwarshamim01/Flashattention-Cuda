#pragma once
#include <cstddef>
#include "config.hpp"
#include "tensor.hpp"

namespace fa {

// bytes to hold (m,l,acc[D]) for [splits,B,H,S]
inline size_t forward_splitk_workspace_bytes(int splits, int B, int H, int S, int D){
  size_t rows = static_cast<size_t>(splits) * B * H * S;
  size_t per_row = (2 + D) * sizeof(float);
  return rows * per_row;
}

} // namespace fa
