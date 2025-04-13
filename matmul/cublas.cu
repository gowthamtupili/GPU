#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cassert>
#include <chrono>
using std::vector;

// #define M 1024  // Number of rows in A and C
// #define N 1024  // Number of columns in B and C
// #define K 1024  // Number of columns in A and rows in B
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


int main() {
    int M = 1 << 10;
    int N = 1 << 10;
    int K = 1 << 10;
    size_t bytes1 = M * (K) * sizeof(float) ;
    size_t bytes2 = K * (N) * sizeof(float) ;
    size_t bytes3 = M * (N) * sizeof(float) ;

    vector<float> h_A(M * K);
    vector<float> h_B(K * N);
    vector<float> h_C(M * N);

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Initialize matrices
    for(int i=0;i<h_A.size();i++) {
        h_A[i] = rand() % 100;
        // h_A[i] = i *2;
        // h_A[i] = 1.0f;
    }
    for(int i=0;i<h_B.size();i++) {
        h_B[i] = rand() % 100;
        // h_B[i] = i * 10;
        // h_B[i] = 1.0f;
    }
    
    float *d_A, *d_B, *d_C;
    float alpha = 1.0f, beta = 0.0f;

    // Variable to measure time.
    cudaEvent_t start, stop, kernelStart, kernelStop;
    float time = 0, kernelTime = 0; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    
    cudaEventRecord(start);

    cudaMalloc(&d_A, bytes1);
    cudaMalloc(&d_B, bytes2);
    cudaMalloc(&d_C, bytes3);

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    
    cudaEventRecord(kernelStart);
    // Note: cuBLAS uses column-major order, so we treat A, B, C as transposed
    // Hence, cublasSgemm(handle, opB, opA, N, M, K, ...) gives C = A * B in row-major
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,       // N x M output
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);
    
    
    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    
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
    verify_result(h_A, h_B, h_C, M, K, N);
    // std::cout << "Result matrix C (row-major):\n";
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j)
    //         std::cout << h_C[i * N + j] << " ";
    //     std::cout << "\n";
    // }

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
