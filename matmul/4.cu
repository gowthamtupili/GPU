#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using std::cout;
using std::vector;

#define WIDTH 4
#if WIDTH == 1
    typedef float floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#endif
const int TS = 32;

void checkCudaError(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
      std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
      exit(EXIT_FAILURE);
  }
}


// Use wider data types
__global__ void myGEMM4(const int M, const int N, const int K,
     floatX* A,
     floatX* B,
     floatX* C) {
    // printf("hi");
    // Thread identifiers
    // const int row = threadIdx.x; // Local row ID (max: TS/WIDTH)
    // const int col = threadIdx.y; // Local col ID (max: TS)
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    const int globalCol = (TS/WIDTH)*blockIdx.x + col; // 0..N/WIDTH
    const int globalRow = TS*blockIdx.y + row; // 0..M

    // Local memory to fit a tile of TS*TS elements of A and B
    __shared__ floatX Asub[TS][TS/WIDTH];
    __shared__ floatX Bsub[TS][TS/WIDTH];
    // Initialise the accumulation registers
    #if WIDTH == 1
    floatX acc = 0.0f;
    #elif WIDTH == 2
    floatX acc = { 0.0f, 0.0f };
    #elif WIDTH == 4
    floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f };
    #endif

    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        // const int tiledRow = (TS/WIDTH)*t + row;
        // const int tiledCol = TS*t + col;
        Asub[row][col] = A[globalRow*(K/WIDTH) + t*(TS/WIDTH) + col];
        Bsub[row][col] = B[t*(N/WIDTH)*TS + row*(N/WIDTH) + globalCol];

        // Synchronise to make sure the tile is loaded
        __syncthreads();


        // Perform the computation for a single tile
        floatX vecA, vecB;
        // float valB;
        float ValA;
        for (int k=0; k<TS/WIDTH; k++) {
            // vecB = Bsub[col][k];
            vecA = Asub[row][k];
            for (int w=0; w<WIDTH; w++) {
                // vecA = Asub[WIDTH*k + w][row];
                vecB = Bsub[WIDTH*k + w][col];
                #if WIDTH == 1
                ValA = vecA;
                acc += vecB * ValA;
                if(row == 0 && col == 0) {
                    // printf(" %f %f %f ",vecA, valB, acc);
                }
                #elif WIDTH == 2
                switch (w) {
                    case 0: ValA = vecA.x; break;
                    case 1: ValA = vecA.y; break;
                }
                acc.x += vecB.x * ValA;
                acc.y += vecB.y * ValA;
                // if(globalCol == 0 && globalRow == 0) {
                //     printf("%d-%f*%f=%f\n",WIDTH*k + w, ValA, vecB.x ,acc.x);
                // }
                #elif WIDTH == 4
                switch (w) {
                    case 0: ValA = vecA.x; break;
                    case 1: ValA = vecA.y; break;
                    case 2: ValA = vecA.z; break;
                    case 3: ValA = vecA.w; break;
                }
                acc.x += vecB.x * ValA;
                acc.y += vecB.y * ValA;
                acc.z += vecB.z * ValA;
                acc.w += vecB.w * ValA;
                #endif
            }
        }

        // Synchronise before loading the next tile
        __syncthreads();
    }

    // Store the final results in C
    // C[globalCol*(M/WIDTH) + globalRow] = acc;
    C[globalRow*(N/WIDTH) + globalCol] = acc;
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

    size_t bytes1 = M * (K) * sizeof(float) ;
    size_t bytes2 = K * (N) * sizeof(float) ;
    size_t bytes3 = M * (N) * sizeof(float) ;

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


    // Variable to measure time.
    cudaEvent_t start, stop, kernelStart, kernelStop;
    float time = 0, kernelTime = 0; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    
    cudaEventRecord(start);

    floatX *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes1);
    cudaMalloc(&d_b, bytes2);
    cudaMalloc(&d_c, bytes3);

    // Copy data to the device
    cudaMemcpy(d_a, reinterpret_cast<floatX*>(h_a.data()), bytes1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, reinterpret_cast<floatX*>(h_b.data()), bytes2, cudaMemcpyHostToDevice);
    checkCudaError(cudaGetLastError(), "Memory Transfer failed");
    // Use dim3 structs for block  and grid dimensions
    dim3 threads(TS/WIDTH, TS);
    dim3 blocks((N / TS), (M / TS));

	cudaEventRecord(kernelStart);

    myGEMM4<<<blocks, threads>>>(M, N, K, d_a, d_b, d_c);

    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);

    // cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
  
  
    // Copy back to the host
    cudaMemcpy(h_c.data(), reinterpret_cast<float*>(d_c), bytes3, cudaMemcpyDeviceToHost);


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