#include "fa/autotune.hpp"
#include "fa/api.hpp"       // <--- FIXED: Changed from forward.hpp to api.hpp
#include <cuda_runtime.h>
#include <functional>       // Needed for std::function

namespace fa {

static float time_one(cudaStream_t stream, std::function<void(cudaStream_t)> fn) {
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, stream);
  fn(stream);
  cudaEventRecord(b, stream);
  cudaEventSynchronize(b);
  float ms=0.f; cudaEventElapsedTime(&ms,a,b);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms;
}

LaunchConfig FlashAttnAutoTuner::get_or_tune(
    int B, int H, int S, int D, bool causal,
    void* q, void* k, void* v, void* o, int max_trials)
{
  AutoKey key{S,D,causal};
  auto it = cache_.find(key);
  if (it != cache_.end()) return it->second;

  // Small candidate set that covers typical regimes
  LaunchConfig candidates[] = {
    {224, 64, 256, 1}, {128, 64, 256, 1}, {224, 32, 256, 1},
    {224, 64, 256, 2}, {128, 64, 128, 1}, {256, 64, 256, 1}
  };

  float best_ms = 1e30f; LaunchConfig best = candidates[0];
  cudaStream_t stream = 0;

  for (int i=0; i< (int)(sizeof(candidates)/sizeof(candidates[0])) && i<max_trials; ++i) {
    LaunchConfig lc = candidates[i];
    float ms = time_one(stream, [&](cudaStream_t s){
      flash_attention_forward<float>( (float*)q,(float*)k,(float*)v,(float*)o,
        B,H,S,D, causal, lc, s);
      cudaStreamSynchronize(s);
    });
    if (ms < best_ms) { best_ms = ms; best = lc; }
  }
  cache_.insert({key,best});
  return best;
}

} // namespace fa