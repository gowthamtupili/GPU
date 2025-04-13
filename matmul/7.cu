#include <algorithm>
#include <cassert>
// #include <cstdio>
// #include <cstdlib>
// #include <cublas_v2.h>
// #include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

using std::cout;
using std::vector;

typedef unsigned int uint32;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

const uint32 BK = 8;
const uint32 TM = 8;
const uint32 TN = 8;
const uint32 BM = 128;
const uint32 BN = 128;


void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
  }


__global__ void myGEMM7(int M, int N, int K, float *A,
                               float *B, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * (BN+5)];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = threadResults[resIdxM * TN + resIdxN] + tmp.x;
      tmp.y = threadResults[resIdxM * TN + resIdxN + 1] + tmp.y;
      tmp.z = threadResults[resIdxM * TN + resIdxN + 2] + tmp.z;
      tmp.w = threadResults[resIdxM * TN + resIdxN + 3] + tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}



// Check result on the CPU
void verify_result(vector<float> &a, vector<float> &b, vector<float> &c, int M, int N, int K) {
    auto start = std::chrono::high_resolution_clock::now();
    // cout << "Matrix C (CPU) \n";
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        // For every element in the row-column pair
        int tmp = 0;
        for (int k = 0; k < N; k++) {
          // Accumulate the partial results
          tmp += a[i * N + k] * b[k * K + j];
        }
        // Check against the CPU result
        // cout << tmp << " ";
        // if(tmp = c[i * K + j]) {
        //     cout << i << " " << j << "\n";
        // }
        assert(tmp == c[i * K + j]);
      }
    //   cout << "\n";
    }
    // Calculate elapsed time
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "CPU Mat Mul Time: " << duration.count() << " ms\n";
}


int main () {

    // Matrix size of 1024 x 1024;
    int M = 1 << 10;
    int N = 1 << 10;
    int K = 1 << 10;

    size_t bytes1 = M * (K) * sizeof(float);
    size_t bytes2 = K * (N) * sizeof(float);
    size_t bytes3 = M * (N) * sizeof(float);

    vector<float> h_a(M * K);
    vector<float> h_b(K * N);
    vector<float> h_c(M * N);


    // Initialize matrices
    for(int i=0;i<h_a.size();i++) {
        h_a[i] = rand() % 100;
        // h_a[i] = i *2;
        // h_a[i] = 1.0f;
    }
    for(int i=0;i<h_b.size();i++) {
        h_b[i] = rand() % 100;
        // h_b[i] = i * 10;
        // h_b[i] = 1.0f;
    }
//   cout << "Matrix A: " << M << "x" << N << "\n";
//   for(int i = 0; i < M; i++) {
//     for(int j = 0; j < N; j++) {
//       cout << h_a[i*N + j] << " ";
//     }
//     cout << "\n";
//   }

//   cout << "Matrix B: " << N << "x" << K << "\n";
//   for(int i = 0; i < N; i++) {
//     for(int j = 0; j < K; j++) {
//       cout << h_b[i*K + j] << " ";
//     }
//     cout << "\n";
//   }

    // Variable to measure time.
    cudaEvent_t start, stop, kernelStart, kernelStop;
    float time = 0, kernelTime = 0; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    
    cudaEventRecord(start);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes1);
    cudaMalloc(&d_b, bytes2);
    cudaMalloc(&d_c, bytes3);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes2, cudaMemcpyHostToDevice);
    checkCudaError(cudaGetLastError(), "Memory Transfer failed");
    // Use dim3 structs for block  and grid dimensions

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));

	cudaEventRecord(kernelStart);

    myGEMM7 <<<gridDim, blockDim>>>(M, N, K, d_a, d_b, d_c);

    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);

    // cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
  
  
    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, bytes3, cudaMemcpyDeviceToHost);
    // cout << "Matrix C(GPU): " << M << "x" << N << "\n";
    // for(int i = 0; i < M; i++) {
    //     for(int j = 0; j < N; j++) {
    //     cout << h_c[i*K + j] << " ";
    //     }
    //     cout << "\n";
    // }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  
    cudaEventElapsedTime(&time, start, stop);
    cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);
    std::cout << "GPU Kernel time: " << kernelTime << " ms" << std::endl;
    std::cout << "GPU Total time (H2D + Kernel + D2H): " << time << " ms" << std::endl;
  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);
    verify_result(h_a, h_b, h_c, M, K, N);

    cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}