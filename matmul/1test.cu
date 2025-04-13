#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using std::cout;
using std::vector;

void checkCudaError(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
      std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
      exit(EXIT_FAILURE);
  }
}



__global__ void matrixMul(int *a, int *b, int *c, int M, int N, int K) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("%d  %d\n", row, col);
  if(row < M && col < K) {
    // Iterate over row, and down column
    c[row * K + col] = 0;
    for (int i = 0; i < N; i++) {
      // Accumulate results for a single element
      c[row * K + col] += a[row * N + i] * b[i * K + col];
    }
  }
}

// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int M, int N, int K) {
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
      assert(tmp == c[i * K + j]);
    }
    // cout << "\n";
  }
  
  // Calculate elapsed time
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = stop - start;
  std::cout << "CPU Mat Mul Time: " << duration.count() << " ms\n";
}

int main() {
  // Matrix size of 1024 x 1024;
  int M = 1 << 5;
  int N = 1 << 5;
  int K = 1 << 5;
  // int N = 512;

  // Size (in bytes) of matrix
  size_t bytes1 = M * N * sizeof(int);
  size_t bytes2 = N * K * sizeof(int);
  size_t bytes3 = M * K * sizeof(int);

  // Host vectors
  vector<int> h_a(M * N);
  vector<int> h_b(N * K);
  vector<int> h_c(M * K);

  // Initialize matrices
  for(int i=0;i<h_a.size();i++) {
    h_a[i] = rand() % 100;
  }
  for(int i=0;i<h_b.size();i++) {
    h_b[i] = rand() % 100;
  }
  // cout << "Matrix A: " << M << "x" << N << "\n";
  // for(int i = 0; i < M; i++) {
  //   for(int j = 0; j < N; j++) {
  //     cout << h_a[i*N + j] << " ";
  //   }
  //   cout << "\n";
  // }

  // cout << "Matrix B: " << N << "x" << K << "\n";
  // for(int i = 0; i < N; i++) {
  //   for(int j = 0; j < K; j++) {
  //     cout << h_b[i*K + j] << " ";
  //   }
  //   cout << "\n";
  // }

    // Variable to measure time.
    cudaEvent_t start, stop, kernelStart, kernelStop;
    float time = 0, kernelTime = 0; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    
    cudaEventRecord(start);

  // Allocate device memory
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes1);
  cudaMalloc(&d_b, bytes2);
  cudaMalloc(&d_c, bytes3);

  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), bytes1, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes2, cudaMemcpyHostToDevice);

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  // int BLOCKS = (N-1)/ THREADS + 1;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS, 1);
  // dim3 threads(2, 2, 1);
  dim3 blocks((K)/THREADS ,(M)/THREADS, 1);
  // dim3 blocks(1 ,2, 1);

	cudaEventRecord(kernelStart);

  // Launch kernel
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, M, N, K);

  cudaEventRecord(kernelStop);
  cudaEventSynchronize(kernelStop);

  // cudaDeviceSynchronize();
  checkCudaError(cudaGetLastError(), "Kernel launch failed");


  cudaDeviceSynchronize();
  checkCudaError(cudaGetLastError(), "Kernel launch failed");


  // Copy back to the host
  cudaMemcpy(h_c.data(), d_c, bytes3, cudaMemcpyDeviceToHost);

  // cout << "Matrix C(GPU): " << M << "x" << K << "\n";
  // for(int i = 0; i < M; i++) {
  //   for(int j = 0; j < K; j++) {
  //     cout << h_c[i*K + j] << " ";
  //   }
  //   cout << "\n";
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

  // Check result
  verify_result(h_a, h_b, h_c, M, N, K);

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}