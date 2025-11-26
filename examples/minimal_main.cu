#include "fa/api.hpp"
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

int main(){
  int B=1,H=2,S=2048,D=64;
  bool causal=true;
  size_t bytes = (size_t)B*H*S*D*sizeof(float);
  float *q,*k,*v,*o; 
  cudaMalloc(&q,bytes); cudaMalloc(&k,bytes); cudaMalloc(&v,bytes); cudaMalloc(&o,bytes);

  std::cout << "Launching Standard Flash Attention..." << std::endl;
  fa::LaunchConfig lc{224,64,256,1};
  fa::flash_attention_forward<float>(q,k,v,o,B,H,S,D,causal,lc,0);

  // Split-K example
  int splits=2;
  size_t ws = fa::forward_splitk_workspace_bytes(splits,B,H,S,D);
  void* workspace; cudaMalloc(&workspace, ws);
  
  std::cout << "Launching Split-K Flash Attention..." << std::endl;
  fa::flash_attention_forward_splitk<float>(q,k,v,o,B,H,S,D,causal,splits,workspace,ws,lc,0);

  // Synchronize to ensure GPU finished before printing "Done"
  cudaDeviceSynchronize();

  cudaFree(workspace); cudaFree(o); cudaFree(v); cudaFree(k); cudaFree(q);
  
  std::cout << "Success! Kernels finished without crashing." << std::endl;
  return 0;
}