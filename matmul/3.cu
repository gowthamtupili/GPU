#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using std::cout;
using std::vector;


#define WPT 2
const int M = (1 << 10);
const int N = (1 << 10);
const int K = (1 << 10);

#define THREADS 32


void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
  

__global__
void matrixMul(int* a, int* b, int* c) {
    // printf("Hello");
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x * WPT + threadIdx.x;

    // Statically allocated shared memory
    __shared__ int s_a[THREADS][THREADS];
    __shared__ int s_b[THREADS][THREADS];

    const unsigned int RTS = THREADS/WPT;
    // const unsigned int RTS = blockDim.x*gridDim.x;
    // int tmp = 0;
    int acc[WPT] = {0};
    // for (int i = 0; i < WPT; i++)
    // {
    //     acc[i] = 0;
    // }
    

    // Sweep tile across Matrix
    for(int i = 0; i < K; i+=(THREADS)) {
        // Load in elements for this tile

        for(int w = 0; w < WPT; w++) {
            s_a[threadIdx.y][threadIdx.x + w * RTS] = a[row * K + i + threadIdx.x + w * RTS];
            s_b[threadIdx.y][threadIdx.x + w * RTS] = b[i * N + threadIdx.y * N + col + w * RTS];
        }
        __syncthreads();
        
        for(int j = 0; j < (THREADS); j++) {
            for(int w = 0; w < WPT; w++) {
                acc[w] += s_a[threadIdx.y][j] * s_b[j][threadIdx.x + w * RTS];
            }
        }
        __syncthreads();
    }
    for(int w = 0; w < WPT; w++) {
        if (row < M && col < N){
            c[row * N + col + w * RTS] = acc[w];
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
    //   cout << "\n";
    }
    // Calculate elapsed time
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "CPU Mat Mul Time: " << duration.count() << " ms\n";
}

int main () {

    size_t bytes1 = M * K * sizeof(int);
    size_t bytes2 = K * N * sizeof(int);
    size_t bytes3 = M * N * sizeof(int);

    vector<int> h_a(M * K);
    vector<int> h_b(K * N);
    vector<int> h_c(M * N);


    // Initialize matrices
    for(int i=0;i<h_a.size();i++) {
        h_a[i] = rand() % 100;
        // h_a[i] = i;
    }
    for(int i=0;i<h_b.size();i++) {
        h_b[i] = rand() % 100;
        // h_b[i] = i;
    }


    // Variable to measure time.
    cudaEvent_t start, stop, kernelStart, kernelStop;
    float time = 0, kernelTime = 0; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    
    cudaEventRecord(start);

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes1);
    cudaMalloc(&d_b, bytes2);
    cudaMalloc(&d_c, bytes3);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes2, cudaMemcpyHostToDevice);

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS/WPT, THREADS);
    // dim3 threads(1, 4);
    dim3 blocks(N/THREADS, M/THREADS);
    // dim3 blocks(2, 1);

	cudaEventRecord(kernelStart);

    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

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