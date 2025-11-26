cmake_minimum_required(VERSION 3.20)
project(flashattn LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
add_library(flashattn STATIC
  src/forward_kernels.cu
  src/forward_splitk.cu
  src/autotune.cu
  src/api.cu
)
target_include_directories(flashattn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(flashattn PRIVATE
  # Uncomment to enable warp-level cp.async:
  # FA_USE_CP_ASYNC
  # Uncomment to experiment with WMMA path:
  # FA_USE_WMMA
)
# Examples
add_executable(minimal examples/minimal_main.cu)
target_link_libraries(minimal PRIVATE flashattn)
