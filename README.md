# FlashAttention‑CUDA (exact, IO‑aware attention)

**FlashAttention** computes exact scaled dot‑product attention while minimizing **global memory (HBM) IO**. Instead of materializing the $(n\times n)$ score/probability matrices, it streams **(K,V)** tiles through on‑chip memory (shared/reg) and uses a numerically‑stable **online softmax** to produce the exact result. This repo provides a clean, modular CUDA implementation with production features:

* ✅ **Exact** forward pass (no approximation) with **online softmax**
* ✅ **Row‑split multi‑CTA per head** for high occupancy
* ✅ **Warp specialization** (producer / consumer warps)
* ✅ **Double buffering** (2‑stage shared‑memory pipeline; optional `cp.async`)
* ✅ **Vectorized 16‑B copies** (`uint4`) with safe fallbacks
* ✅ **BF16 / FP16 / FP32 I/O**, always **FP32 accumulation**
* ✅ **Split‑K forward** (optional, two‑pass merge)
* ✅ **Autotuner** to pick tile sizes and block configs per ((S,D))
* 🔧 WMMA/Tensor‑Core hooks (optional, off by default)

---

## Table of contents

1. [Quick start](#quick-start)
2. [Public API](#public-api)
3. [Mathematics](#mathematics)

   * [Vanilla attention](#vanilla-attention)
   * [Numerically stable softmax](#numerically-stable-softmax)
   * [Online (streaming) softmax derivation](#online-streaming-softmax-derivation)
   * [Causal & padding masks](#causal--padding-masks)
   * [IO & complexity](#io--complexity)
   * [Numerical precision](#numerical-precision)
4. [Algorithm & pseudocode](#algorithm--pseudocode)
5. [CUDA design](#cuda-design)

   * [Data layout & grid mapping](#data-layout--grid-mapping)
   * [Warp specialization](#warp-specialization)
   * [Double buffering](#double-buffering)
   * [Vectorized & async copies](#vectorized--async-copies)
   * [Row‑split multi‑CTA per head](#rowsplit-multi-cta-per-head)
   * [Split‑K forward (two‑pass)](#splitk-forward-two-pass)
   * [Tensor Cores (WMMA) hook](#tensor-cores-wmma-hook)
   * [Shared memory & registers](#shared-memory--registers)
6. [File structure](#file-structure)
7. [Build & flags](#build--flags)
8. [Tuning guide](#tuning-guide)
9. [Validation & tests](#validation--tests)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## Quick start

```bash
# Configure & build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# (Optional) if you added the example target in CMake:
./build/minimal
```

**Architectures**: SM70+ runs the fallback path; SM80+/SM90 get async copy when `FA_USE_CP_ASYNC` is enabled (see flags).
**Layout**: All tensors are contiguous `[B, H, S, D]`.

---

## Public API

Header: `include/fa/api.hpp`

```cpp
namespace fa {

// Simple forward (row‑split multi‑CTA per head)
template<typename T>   // T = float, __half, __nv_bfloat16
void flash_attention_forward(
  T* q, T* k, T* v, T* o,
  int B, int H, int S, int D,
  bool causal,
  const LaunchConfig& lcfg,
  cudaStream_t stream = 0);

// Split‑K forward (two‑pass merge)
template<typename T>
void flash_attention_forward_splitk(
  T* q, T* k, T* v, T* o,
  int B, int H, int S, int D,
  bool causal,
  int k_splits,                     // >1 enables split‑K
  void* workspace, size_t workspace_bytes,
  const LaunchConfig& lcfg,
  cudaStream_t stream = 0);

struct LaunchConfig {
  int tile_m = 224;   // queries per CTA
  int tile_n = 64;    // keys per tile
  int block  = 256;   // threads per CTA
  int loaders= 1;     // producer warps
};

} // namespace fa
```

**Autotuner** (optional): `include/fa/autotune.hpp`

```cpp
fa::FlashAttnAutoTuner tuner;
auto best = tuner.get_or_tune(B,H,S,D, causal, q,k,v,o);
// then call flash_attention_forward<T>(..., best, stream);
```

---

## Mathematics

### Vanilla attention

For a single head with (Q\in\mathbb{R}^{n\times d}), (K\in\mathbb{R}^{n\times d}), (V\in\mathbb{R}^{n\times d_v}) and scale (\alpha = 1/\sqrt{d}):

[
\mathrm{Attn}(Q,K,V)
= \mathrm{softmax}\big(\alpha, QK^\top\big),V
\quad\in\mathbb{R}^{n\times d_v}.
]

A naïve implementation materializes (S=\alpha QK^\top\in\mathbb{R}^{n\times n}), applies row‑wise softmax to get (P), then outputs (O=PV). The memory footprint and traffic of (S) and (P) are **quadratic in (n)**.

### Numerically stable softmax

For a row (x\in\mathbb{R}^m), the stable softmax rescales by the max:

[
\mathrm{softmax}(x)_j
= \frac{e^{x_j - m}}{\sum_k e^{x_k - m}},\quad m=\max_j x_j.
]

Subtracting (m) avoids overflow/underflow.

### Online (streaming) softmax derivation

We want the softmax output
[
o_i = \sum_{j=1}^n p_{ij},v_j,
\quad p_{ij} = \frac{e^{s_{ij}}}{\sum_{k=1}^n e^{s_{ik}}},
\quad s_{ij} = \alpha, q_i^\top k_j.
]

Process keys in tiles (T_1, T_2, \dots, T_t). For a fixed query row (i), maintain:

* running max (m),
* running denominator (\ell = \sum_{j\in \text{seen}} e^{s_{ij}-m}),
* running numerator vector (\mathbf{a} = \sum_{j\in \text{seen}} e^{s_{ij}-m}, v_j).

**Derivation for merging a new tile (T):**

Let (m_T=\max_{j\in T} s_{ij}) and (m'=\max(m, m_T)). Then
[
\begin{aligned}
\ell' &=
\sum_{j\in \text{seen}} e^{s_{ij}-m'}

* \sum_{j\in T} e^{s_{ij}-m'} \
  &= e^{m-m'} \underbrace{\sum_{j\in \text{seen}} e^{s_{ij}-m}}_{\ell}
* \sum_{j\in T} e^{s_{ij}-m'},
  \
  \mathbf{a}' &=
  e^{m-m'} \mathbf{a}
* \sum_{j\in T} e^{s_{ij}-m'}, v_j.
  \end{aligned}
  ]

This is the **online/streaming softmax** update. After all tiles:
[
o_i = \frac{\mathbf{a}}{\ell}.
]
It is **exact** (identical to full softmax), because we merely change the reference (m) via algebraic rescaling.

### Causal & padding masks

* **Causal**: set logits for (j>i) to (-\infty). In streaming, simply **skip** masked keys, i.e., do not update (\ell) or (\mathbf{a}) for them.
* **Padding**: if valid key length is (L), ignore (j\ge L).

If a full tile is masked, no updates occur; state ((m,\ell,\mathbf{a})) remains unchanged.

### IO & complexity

* **Arithmetic** stays (O(n d^2)) (per head) up to constants (dot products & accumulations).
* **HBM IO** is the bottleneck at large (n). FlashAttention never writes/intermediates (S) or (P); it streams (K,V) and keeps partials in **shared/registers**. For a single head:

  * Read (Q) once per row, read (K,V) once per tile, write (O) once.
  * IO roughly (O(n d + n d_v)) rather than (O(n^2)) intermediates.
* **Row‑split multi‑CTA** increases parallelism: multiple CTAs per head re‑read (K,V) but L2 helps amortize; the occupancy win typically dominates.

### Numerical precision

* Inputs (Q,K,V) may be **FP16/BF16/FP32**; **accumulation is FP32**.
* Use (\alpha = 1/\sqrt{d}) scaling to keep logits in a well‑behaved range.
* Online softmax keeps exponents centered via the running max (m), curbing overflow/underflow.
* If (\ell=0) (fully masked row), we emit zeros.

---

## Algorithm & pseudocode

For each batch (b), head (h), and query row (i), we process (K,V) in tiles:

```text
Initialize: m = -inf, l = 0, a[:] = 0

for key tile T = {j = k0 .. k0+N-1}:
  mT = max_j_in_T ( α * dot(q[i,:], K[j,:]) )   // skip masked j
  m_new = max(m, mT)
  scale = exp(m - m_new)
  l *= scale;  a[:] *= scale

  for j in T (skip masked):
    s = exp( α*dot(q[i],K[j]) - m_new )
    l += s
    a[:] += s * V[j,:]
  m = m_new

o[i,:] = a[:] / l
```

We map each **query row to a consumer thread**; (K,V) tiles are cooperatively loaded by **producer warps** into shared memory (double‑buffered).

---

## CUDA design

### Data layout & grid mapping

* Tensors are contiguous `[B,H,S,D]`.
* **Row‑split** over (S): `grid.x = ceil(S / tile_m)`, `grid.y = H`, `grid.z = B`.
* One CTA handles `tile_m` query rows. Within a CTA:

  * `loader_warps` (default 1) are **producers**.
  * The remaining threads are **consumers**; each consumer thread owns **one query row**.

### Warp specialization

* **Producer warp(s)** only prefetch the **next** (K,V) tile from HBM to shared memory.
* **Consumer warps** compute dot‑products and online softmax updates on the **current** tile.
* This reduces barriers and hides memory latency versus “everyone does everything.”

### Double buffering

We keep two shared‑memory stages for each of (K) and (V):

```
Iteration t:   compute(K/V stage A)   | prefetch next K/V -> stage B
Barrier+swap
Iteration t+1: compute(K/V stage B)   | prefetch next K/V -> stage A
```

Barriers only at tile boundaries. On SM80+, enable `FA_USE_CP_ASYNC` to issue `cp.async` lines for genuine async copies; otherwise, we still overlap via warp scheduling.

### Vectorized & async copies

* If pointers are **16‑B aligned** and the byte count is a multiple of 16, we use **`uint4`** (16‑B) copies cooperatively across lanes.
* Else, we fall back to scalar loads.
* With `FA_USE_CP_ASYNC` (SM80+), the producer warp emits **`cp.async`** into shared memory for further latency hiding. The code has both paths.

**SMEM footprint** (row‑split kernel):
We keep two stages of (K) and two of (V) in shared memory:

[
\text{SMEM bytes} = 4 \cdot \text{tile_n} \cdot D \cdot \mathrm{sizeof}(\mathrm{Storage}),
]
where `Storage` is the input type (`float`/`half`/`bfloat16`). Choose `tile_n` to fit your device SMEM.

### Row‑split multi‑CTA per head

* `grid.x` splits the **query rows** across CTAs for the same head.
* Forward pass needs **no cross‑CTA reductions** (each row’s softmax is independent).
* This boosts occupancy especially when (B\cdot H) is small and (S) is large.

### Split‑K forward (two‑pass)

When you also want to split along **keys** (e.g., keep CTAs smaller or match cache limits):

1. **Partial pass**: launch (S_K) splits; each split processes a disjoint key range ([k_{\text{begin}},k_{\text{end}})). It outputs **per‑row partials** ((m^{(t)}, \ell^{(t)}, \mathbf{a}^{(t)})) into a workspace (FP32).
2. **Merge pass**: per row, merge all splits via the same online‑softmax merge rule:

[
\begin{aligned}
m &= \max_t m^{(t)},\
\ell &= \sum_t e^{m^{(t)}-m},\ell^{(t)},\quad
\mathbf{a} = \sum_t e^{m^{(t)}-m},\mathbf{a}^{(t)},\
\mathbf{o} &= \mathbf{a}/\ell.
\end{aligned}
]

**Workspace size** for (S_K) splits:

[
\mathrm{bytes} = S_K \cdot B \cdot H \cdot S \cdot \big(2 + D\big)\cdot \mathrm{sizeof(float)}.
]

Row‑split and split‑K can be combined if needed (row‑split for parallelism; split‑K for memory/capacity).

### Tensor Cores (WMMA) hook

The default inner products are scalar FP32 for clarity and portability. A WMMA path can tile to **16×16×16** fragments (HMMA) and still use the online softmax. Requirements:

* (D) multiple of 16, data in `half`/`bfloat16`.
* Use fragments for (Q) and (K^\top) sub‑tiles, accumulate in FP32.
* Keep the **two‑pass per‑tile** structure (max pass then sum pass) with the same rescaling.

This repo includes compile‑time hooks (`FA_USE_WMMA`) so you can drop in a WMMA consumer later.

### Shared memory & registers

* Shared memory holds **two (K) tiles + two (V) tiles**.
* Each consumer thread keeps its **(q) row**, ((m,\ell)), and the **accumulator** (\mathbf{a}) in registers (FP32).
* `tile_m` is chosen so the number of consumer threads roughly equals `tile_m` (one row per consumer thread).

---

## File structure

```
include/fa/
  config.hpp        # tile sizes, warp size, KernelConfig/LaunchConfig
  traits.hpp        # TypeTraits<T> with float/half/bfloat16 I/O and FP32 convert
  tensor.hpp        # contiguous [B,H,S,D] Tensor4D view
  softmax.hpp       # online softmax state (m, l) + rescale
  tile_loader.hpp   # producer warp: 16B vectorized copies + optional cp.async
  row_compute.hpp   # consumer: per-row dot products + online softmax update
  forward_kernels.cuh  # kernel declarations (row-split, split-K partial/merge)
  workspace.hpp     # workspace sizing helpers (split-K)
  autotune.hpp      # tiny autotuner class
  api.hpp           # user-facing API (forward / split-K forward)

src/
  forward_kernels.cu   # row-split kernel (multi-CTA per head, WS + double-buffer)
  forward_splitk.cu    # split-K partial + merge kernels (two-pass)
  autotune.cu          # runtime tuner (tries small config grid, caches best)
  api.cu               # API implementations + kernel launch plumbing

examples/
  minimal_main.cu      # (optional) tiny smoke test
```

---

## Build & flags

### CMake (excerpt)

```cmake
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)   # tune for your GPUs

add_library(flashattn STATIC
  src/forward_kernels.cu
  src/forward_splitk.cu
  src/autotune.cu
  src/api.cu
)
target_include_directories(flashattn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(flashattn PRIVATE
  # Enable warp-level cp.async on SM80+ (optional):
  # FA_USE_CP_ASYNC
  # Enable WMMA/Tensor Cores path (optional, experimental hook):
  # FA_USE_WMMA
)
```

**Dynamic shared‑memory opt‑in** is handled in the API before launches via `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)` and setting carve‑out preference to 100%.

---

## Tuning guide

* **SMEM sizing** (row‑split kernel):
  (\text{SMEM} = 4 \times \text{tile_n} \times D \times \mathrm{sizeof}(\text{Storage})).
  For FP16/BF16 Storage and `tile_n=64`, `D=128` → (4×64×128×2 = 65{,}536) bytes.

* **tile_m** ≈ number of consumer threads = `block_threads - 32*loader_warps`.
  Default: `256 - 32 = 224` → `tile_m=224` (one row per consumer thread).

* **loader_warps**: start at 1; try 2 if memory‑bound.

* **Autotuner**: `FlashAttnAutoTuner::get_or_tune(...)` probes a small set and caches per ((S,D,\text{causal})).

* **Split‑K**: For very long sequences or tight SMEM, try `k_splits=2..4`. Remember the workspace size.

* **WMMA**: When enabling `FA_USE_WMMA`, ensure (D) multiple of 16 and sufficient registers; measure carefully.

---

## Validation & tests

**Correctness (sanity)**

* Compare the kernel with a **naïve CPU** or small **GPU** reference on random inputs for small (B,H,S,D). Expect max relative error on the order of FP32 rounding (or FP16/BF16 conversion error if using those types).

**Stress cases**

* Very long (S) (e.g., (S\ge 16\mathrm{k})) to ensure online softmax stability.
* All‑masked rows (causal first row, or padding length zero).
* Mixed precision: inputs in FP16/BF16, verify against FP32 reference.

---

## Troubleshooting

* **`undefined reference` at link**
  Ensure you compile and link all four source files (`forward_kernels.cu`, `forward_splitk.cu`, `autotune.cu`, `api.cu`). Check that your app links against `flashattn`.

* **`no such file ./build/minimal`**
  Add the example target to `CMakeLists.txt`:

  ```cmake
  add_executable(minimal examples/minimal_main.cu)
  target_link_libraries(minimal PRIVATE flashattn)
  ```

* **`std::function` error in `autotune.cu`**
  Include `<functional>` or use the templated `time_ms` variant (we provide a version that avoids `std::function`).

* **`too much shared memory`**
  Reduce `tile_n` or `D`, or switch input Storage to FP16/BF16. Also make sure the kernel opts in to large dynamic SMEM (we do this in `api.cu`).

* **Under‑utilization on small (B\cdot H)**
  Increase `grid.x` by decreasing `tile_m` (more CTAs) or enable **split‑K**.

---

## References

* **FlashAttention: Fast and Memory‑Efficient Exact Attention with IO‑Awareness**, Tri Dao et al., NeurIPS 2022.
* **FlashAttention‑2: Faster Attention with Improved Parallelism and Work Partitioning**, Tri Dao et al., 2023.
* **From Online Softmax to FlashAttention**, Zihao Ye, explanatory note.

---

### Appendix: computing notes & derivations (expanded)

**Why the online merge is exact**

Given two disjoint key sets (A,B), with per‑row maxima (m_A, m_B), let
[
\ell_A=\sum_{j\in A} e^{s_j - m_A},\quad
\mathbf{a}*A = \sum*{j\in A} e^{s_j - m_A} v_j,
]
and similarly for (B). For (m'=\max(m_A,m_B)),
[
\sum_{j\in A\cup B} e^{s_j}
= e^{m'}!\left(e^{m_A-m'}\ell_A + e^{m_B-m'}\ell_B\right),
]
and
[
\sum_{j\in A\cup B} e^{s_j} v_j
= e^{m'}!\left(e^{m_A-m'}\mathbf{a}_A + e^{m_B-m'}\mathbf{a}_B\right).
]
Thus the merged state is
[
\ell' = e^{m_A-m'}\ell_A + e^{m_B-m'}\ell_B,\quad
\mathbf{a}' = e^{m_A-m'}\mathbf{a}_A + e^{m_B-m'}\mathbf{a}_B,
]
and the final output (\mathbf{o}=\mathbf{a}'/\ell') equals the full softmax result. Streaming over tiles applies the same algebra iteratively.

**HBM traffic sketch**

* Naïve materialization: write/read (S\in\mathbb{R}^{n\times n}) and (P\in\mathbb{R}^{n\times n}) → (O(n^2)) IO.
* FlashAttention: read (Q) once, stream (K,V) in tiles (read once), write (O) once → (O(nd) + O(nd_v)) IO. Arithmetic is unchanged; performance scales with IO reduction.

**Numerics**

* With FP16/BF16 inputs, convert to FP32 for accumulation, keep the log‑sum‑exp reference (m), and scale contributions by (\exp(s-m)). This keeps intermediate magnitudes (\mathcal{O}(1)) and avoids overflow (e.g., (e^{80}) in FP32 is already huge without stabilization).

---

