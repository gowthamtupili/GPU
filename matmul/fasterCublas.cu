#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cublasLt.h>
#include <chrono>
#include <cassert>

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
        // Matrix size of 1024 x 1024;
    int M = 1 << 10;
    int N = 1 << 10;
    int K = 1 << 10;
    
    size_t bytes1 = M * (K) * sizeof(float) ;
    size_t bytes2 = K * (N) * sizeof(float) ;
    size_t bytes3 = M * (N) * sizeof(float) ;

    vector<float> h_A(M * K);
    vector<float> h_B(K * N);
    vector<float> h_C(M * N);

        // Initialize matrices
    for(int i=0;i<h_A.size();i++) {
        h_A[i] = rand() % 10;
        // h_a[i] = i *2;
        h_A[i] = 1.0f;
    }
    for(int i=0;i<h_B.size();i++) {
        h_B[i] = rand() % 10;
        // h_b[i] = i * 10;
        h_B[i] = 1.0f;
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

    // Create cuBLASLt handle
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);
    // Matrix descriptors
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtMatmulDesc_t matmulDesc;

    // cublasOperation_t opTranspose = CUBLAS_OP_N;

    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, M, K, K);
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, K, N, N);
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, M, N, N);

    cudaEventRecord(kernelStart);

    cublasLtMatmul(ltHandle,
        matmulDesc,
        &alpha,
        d_A, layoutA,
        d_B, layoutB,
        &beta,
        d_C, layoutC,
        d_C, layoutC,
        nullptr, nullptr, 0, 0);


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
    // std::cout << "Result matrix C (row-major):\n";
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j)
    //         std::cout << h_C[i * N + j] << " ";
    //     std::cout << "\n";
    // }
    verify_result(h_A, h_B, h_C, M, K, N);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtDestroy(ltHandle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
