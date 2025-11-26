#pragma once
#include <cstddef>
#include <cstdint>

namespace fa {

template<typename T>
class Tensor4D {
public:
  Tensor4D() : data_(nullptr), B_(0), H_(0), S_(0), D_(0) {}
  Tensor4D(T* data, int B, int H, int S, int D)
    : data_(data), B_(B), H_(H), S_(S), D_(D) {}

  __host__ __device__ T* data() const { return data_; }
  __host__ __device__ int B() const { return B_; }
  __host__ __device__ int H() const { return H_; }
  __host__ __device__ int S() const { return S_; }
  __host__ __device__ int D() const { return D_; }

  __host__ __device__ size_t idx(int b, int h, int s, int d) const {
    return (((size_t)b * H_ + h) * S_ + s) * D_ + d;
  }

  __host__ __device__ T& operator()(int b, int h, int s, int d) const {
    return data_[idx(b,h,s,d)];
  }

  __host__ __device__ T* row_ptr(int b, int h, int s) const {
    return data_ + idx(b,h,s,0);
  }

private:
  T* data_;
  int B_, H_, S_, D_;
};

} // namespace fa
