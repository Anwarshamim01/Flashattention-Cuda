#pragma once
#include <unordered_map>
#include <tuple>
#include "api.hpp"

namespace fa {

struct AutoKey { int S, D; bool causal; };
struct AutoKeyHash {
  size_t operator()(const AutoKey& k) const {
    return std::hash<int>()(k.S) ^ (std::hash<int>()(k.D)<<1) ^ (k.causal?0x9e3779b9:0x7f4a7c15);
  }
};
inline bool operator==(const AutoKey& a, const AutoKey& b){ return a.S==b.S && a.D==b.D && a.causal==b.causal; }

class FlashAttnAutoTuner {
  std::unordered_map<AutoKey, LaunchConfig, AutoKeyHash> cache_;
public:
  LaunchConfig get_or_tune(int B, int H, int S, int D, bool causal,
                           void* q, void* k, void* v, void* o, int max_trials=6);
};

} // namespace fa
