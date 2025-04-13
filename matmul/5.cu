// code for 5th kernel as of now
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
 
 
// using namespace std;
using std::cout;
using std::vector;
 
// #define TSM 32    // Tile-size in dimension M
// #define TSN 32    // Tile-size in dimension N
// #define TSK 32    // Tile-size in dimension K
// #define WPTN 8    // Work-per-thread in dimension N
#define WPT 4
// #define RTSN (TSN/WPTN) // Reduced tile-size in N
// #define LPT ((TSK*TSM)/(TSM*RTSN)) // Loads-per-thread per tile
 
// For Transpose kernel
#define THREADSX 16
#define THREADSY 16
#define THREADS 16
 
#define TRANSPOSEX 4
#define TRANSPOSEY 4

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
 
// Simple transpose kernel for a P * Q matrix
__global__ void transpose(int P, int Q, const float* input, float* output) {
    // Thread identifiers
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ID0 = blockIdx.x * TRANSPOSEX + tx; // 0..Q
    const int ID1 = blockIdx.y * TRANSPOSEY + ty; // 0..P
 
    // Shared memory for efficient transpose
    __shared__ float buffer[TRANSPOSEY][TRANSPOSEX];
 
    // Read from global memory (row-major)
    if (ID1 < P && ID0 < Q) {
        buffer[ty][tx] = input[ID1 * Q + ID0];
    }
   
    __syncthreads();
   
    // Compute new indices after transpose
    const int newID0 = blockIdx.y * TRANSPOSEY + tx; // 0..P
    const int newID1 = blockIdx.x * TRANSPOSEX + ty; // 0..Q
   
    // Write to global memory (row-major)
    if (newID0 < P && newID1 < Q) {
        output[newID1 * P + newID0] = buffer[tx][ty];
    }
}
 
 
// Check result on the CPU for matrix transpose
void verify_transpose(vector<float> &a, vector<float> &b, int M, int N) {
  auto start = std::chrono::high_resolution_clock::now();
  // cout << "Transposed Matrix B (CPU) \n";
  for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
          // Verify that b[j * M + i] is the transpose of a[i * N + j]
          assert(b[j * M + i] == a[i * N + j]);
      }
  }
  // Calculate elapsed time
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = stop - start;
  std::cout << "CPU Time: " << duration.count() << " ms\n";
}
 
__global__ void myGEMM5(const int M, const int N, const int K, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x * WPT + threadIdx.x;
    // printf("h");
    // Statically allocated shared memory
    __shared__ float s_a[THREADSY][THREADSX];
    __shared__ float s_b[THREADSY][THREADSX+2];
    __shared__ int indexCalc;
 
    const unsigned int RTS = THREADSX/WPT;
    // const unsigned int RTS = blockDim.x*gridDim.x;
    // int tmp = 0;
    float acc[WPT];
    for (int i = 0; i < WPT; i++)
    {
        acc[i] = 0.0f;
    }
   
    if(threadIdx.y == 0 && threadIdx.x == 0) indexCalc = col*K;
    __syncthreads();
    // Sweep tile across Matrix
    for(int i = 0; i < K; i+=(THREADSX)) {
        // Load in elements for this tile
        // printf("h");
        for(int w = 0; w < WPT; w++) {
           
            s_a[threadIdx.y][threadIdx.x + w * RTS] = a[row * K + i + threadIdx.x + w * RTS];
            // s_b[threadIdx.y][threadIdx.x + w * RTS] = b[row * K + i + threadIdx.x + w * RTS];
            s_b[threadIdx.y][threadIdx.x + w * RTS] = b[indexCalc + threadIdx.y * K + i + threadIdx.x + w * RTS];
            // if(row == 4 && col == 5) {
            //   // printf("%d %d %d %d %d\n", r, threadIdx.y * K, i, threadIdx.x, w * RTS);
            // }
        }
        __syncthreads();
        // if(row == 0 && col == 0) {
        //     printf("\n\n\n");
        //   for(int i=0;i<THREADS;i++) {
        //     for(int j=0;j<THREADS;j++) {
        //       printf("%f ", s_a[i][j]);
        //     }
        //     printf("\n");
        //   }
        //   printf("\n");
        //   for(int i=0;i<THREADS;i++) {
        //     for(int j=0;j<THREADS;j++) {
        //       printf("%f ", s_b[i][j]);
        //     }
        //     printf("\n");
        //   }
        // }
        for(int j = 0; j < (THREADSX); j++) {
            for(int w = 0; w < WPT; w++) {
                acc[w] += s_a[threadIdx.y][j] * s_b[threadIdx.x+ w*RTS][j];
                // if(row == 64 && col == 0)
                //   printf("\n%f*%f=%f\n",s_a[threadIdx.y][j], s_b[threadIdx.x][j],acc[w]);
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
void verify_result(vector<float> &a, vector<float> &b, vector<float> &c, int M, int N, int K) {
    auto start = std::chrono::high_resolution_clock::now();
    // cout << "Matrix C (CPU) \n";
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        // For every element in the row-column pair
        float tmp = 0;
        for (int k = 0; k < K; k++) {
          // Accumulate the partial results
          tmp += a[i * K + k] * b[j * K + k];
        }
        // Check against the CPU result
        // cout << tmp << " ";
        // if(tmp != c[i * N + j]) {
        //   cout << "I: " << i << " J: " << j << "\n";
        //   cout << " Temp: " << tmp << " Val: " << c[i * K + j] << "\n";
        //   cout << " Diff: " << tmp-c[i * N + j] << "\n";
        // }
        assert(tmp == c[i * N + j]);
      }
    //   cout << "\n";
    }
    // Calculate elapsed time
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "CPU Mat Mul Time: " << duration.count() << " ms\n";
}
 
 
// // Check result on the CPU
// void verify_result(vector<float> &a, vector<float> &b, vector<float> &c, int M, int N, int K) {
//     auto start = std::chrono::high_resolution_clock::now();
//     cout << "Matrix C (CPU) \n";
//     // int count = 0;
//     for (int i = 0; i < M; i++) {
//       for (int j = 0; j < K; j++) {
//         // For every element in the row-column pair
//         float tmp = 0;
//         for (int k = 0; k < N; k++) {
//           // Accumulate the partial results
//           tmp += a[i * N + k] * b[k * K + j];
//         //   if(i == 16 && j == 0) {
//         //     cout << a[i * N + k] << " " << b[k * K + j] << "\n";
//         //     cout << "Tmp: " << tmp << "\n";
//         //   }
//         }
//         // Check against the CPU result
//         // cout << tmp << " ";
//         // if(tmp != c[i * K + j]) {
//         //   cout << i << " " << j << "\n";
//         // }
//         assert(tmp == c[i * K + j]);
       
//         // if(tmp != c[i * K + j]) {
//         //   cout << "I: " << i << " J: " << j << "\n";
//         //   cout << " Temp: " << tmp << " Val: " << c[i * K + j] << "\n";
//         //   cout << " Diff: " << tmp-c[i * K + j] << "\n";
//         //   cout << " Count: " << count << "\n";
//         //   count = count + 1;
//         // }
//       }
//       // cout << "\n";
//     }
//     // Calculate elapsed time
//     auto stop = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> duration = stop - start;
//     std::cout << "CPU Mat Mul Time: " << duration.count() << " ms\n";
// }
 
int main() {
    // Matrix dimensions
    int M = 1 << 10;
    int K = 1 << 10;
    int N = 1 << 10;
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
 
    vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0), h_BT(N * K);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = rand() % 100;
        // h_A[i] = i;
        // h_A[i] = 1.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = rand() % 100;
        // h_B[i] = i;
        // h_B[i] = 1.0f;

    }
   
//   cout << "Matrix A: " << M << "x" << K << "\n";
//   for(int i = 0; i < M; i++) {
//     for(int j = 0; j < N; j++) {
//       cout << h_A[i*N + j] << " ";
//     }
//     cout << "\n";
//   }
 
//   cout << "Matrix B: " << K << "x" << N << "\n";
//   for(int i = 0; i < N; i++) {
//     for(int j = 0; j < K; j++) {
//       cout << h_B[i*K + j] << " ";
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

 
    float *d_A, *d_B, *d_BT, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_BT, bytesB);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);
 
    cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_BT, h_B.data(), bytesB, cudaMemcpyHostToDevice);
 
    // dim3 threadsT(THREADSY, THREADSX);
    // dim3 blocksT(N/THREADSY, K/THREADSX);
    dim3 threadsT(TRANSPOSEX, TRANSPOSEY);
    dim3 blocksT((N + TRANSPOSEX - 1) / TRANSPOSEX, (K + TRANSPOSEY - 1) / TRANSPOSEY);
    transpose<<<blocksT, threadsT>>>(K, N, d_BT, d_B);
    cudaMemcpy(h_BT.data(), d_B, bytesB, cudaMemcpyDeviceToHost);
    verify_transpose(h_B, h_BT, K, N);
    // cout << "Tranpose B: " << K << "x" << N << "\n";
    // for(int i = 0; i < N; i++) {
    //   for(int j = 0; j < K; j++) {
    //     cout << h_BT[i*K + j] << " ";
    //   }
    //   cout << "\n";
    // }
    // dim3 threads(TSM, RTSN);
    // dim3 blocks(M / TSM, N / TSN);
 
    // dim3 threads(THREADS/WPT, THREADS);
    // dim3 blocks(N/THREADS, M/THREADS);

    dim3 threads(THREADSX/WPT, THREADSY);
    dim3 blocks(N/THREADSX, M/THREADSY);
 
	cudaEventRecord(kernelStart);

    myGEMM5<<<blocks, threads>>>(M, N, K, d_A, d_B, d_C);
   
    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);
    // cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
 
    cudaMemcpy(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost);
    // cout << "Matrix C(GPU): " << M << "x" << N << "\n";
    // for(int i = 0; i < M; i++) {
    //     for(int j = 0; j < N; j++) {
    //     cout << h_C[i*K + j] << " ";
    //     }
    //     cout << "\n";
    // }
    cudaMemcpy(h_B.data(), d_B, bytesB, cudaMemcpyDeviceToHost);
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

    verify_result(h_A, h_B, h_C, M, N, K);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
 