#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>


using std::cout;
using std::vector;



void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}



#define TRANSPOSEX 16
#define TRANSPOSEY 16

__global__ void transpose(int P, int Q, const int* input, int* output) {
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

// Kernel launch example:
// dim3 blockSize(TRANSPOSEX, TRANSPOSEY);
// dim3 gridSize((Q + TRANSPOSEX - 1) / TRANSPOSEX, (P + TRANSPOSEY - 1) / TRANSPOSEY);
// transpose<<<gridSize, blockSize>>>(P, Q, d_input, d_output);



// Check result on the CPU for matrix transpose
void verify_transpose(vector<int> &a, vector<int> &b, int M, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    // cout << "Transposed Matrix B (CPU) \n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Verify that b[j * M + i] is the transpose of a[i * N + j]
            assert(b[j * M + i] == a[i * N + j]);
        }
    }
}


int main () {
    int M = 1 << 10;
    int N = 1 << 10;
    size_t bytes1 = M * N * sizeof(int);

    vector<int> h_a(M * N);
    vector<int> h_b(N * M);


    // Initialize matrices
    for(int i=0;i<h_a.size();i++) {
        h_a[i] = i;
    }

    // Variable to measure time.
    cudaEvent_t start, stop, kernelStart, kernelStop;
    float time = 0, kernelTime = 0; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    
    cudaEventRecord(start);

    int *d_a, *d_b;
    cudaMalloc(&d_a, bytes1);
    cudaMalloc(&d_b, bytes1);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes1, cudaMemcpyHostToDevice);

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(TRANSPOSEX, TRANSPOSEY);
    // dim3 threads(1, 4);
    dim3 blocks((N + TRANSPOSEX - 1) / TRANSPOSEX, (M + TRANSPOSEY - 1) / TRANSPOSEY);
    // dim3 blocks(2, 1);

	cudaEventRecord(kernelStart);

    transpose<<<blocks, threads>>>(M, N, d_a, d_b);

    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
  
    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);
  
    // Copy back to the host
    cudaMemcpy(h_b.data(), d_b, bytes1, cudaMemcpyDeviceToHost);
    // cout << "Matrix A(GPU): " << M << "x" << N << "\n";
    // for(int i = 0; i < M; i++) {
    //     for(int j = 0; j < N; j++) {
    //     cout << h_a[i*N + j] << " ";
    //     }
    //     cout << "\n";
    // }
    // cout << "Matrix B(GPU): " << N << "x" << M << "\n";
    // for(int i = 0; i < N; i++) {
    //     for(int j = 0; j < M; j++) {
    //     cout << h_b[i*M + j] << " ";
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
    verify_transpose(h_a, h_b, M, N);

    cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    // cudaFree(d_c);

    return 0;
}