// Using Shared Memory

#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>


using std::cout;
using std::vector;


#define THREADS 256
#define WPT 8

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
  


__global__ void histogram_vectorized(char* input, int* output, int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tx = threadIdx.x;
    __shared__ unsigned int SM[26];

    // Initialize shared memory histogram
    if (tx < 26)
        SM[tx] = 0;
    __syncthreads();

    // Vectorized processing (4 chars per int)
    int* input4 = reinterpret_cast<int*>(input);
    int total_threads = gridDim.x * blockDim.x;
    int n_vec = N / 4;

    // ----------------------------------------------------------------------------------------------
    // unrolling the loop using pragma
    #pragma unroll
    // ----------------------------------------------------------------------------------------------


    for (int i = tid; i < n_vec; i += total_threads) {
        int val = input4[i];

        // Extract 4 chars from the int
        char c0 = (val >> 0)  & 0xFF;
        char c1 = (val >> 8)  & 0xFF;
        char c2 = (val >> 16) & 0xFF;
        char c3 = (val >> 24) & 0xFF;

        atomicAdd(&SM[c0 - 'A'], 1);
        atomicAdd(&SM[c1 - 'A'], 1);
        atomicAdd(&SM[c2 - 'A'], 1);
        atomicAdd(&SM[c3 - 'A'], 1);
    }

    // Handle remaining characters (tail)
    int tail_start = n_vec * 4;
    for (int i = tid + tail_start; i < N; i += total_threads) {
        char val = input[i];
        atomicAdd(&SM[val - 'A'], 1);
    }

    __syncthreads();

    // Global atomic update
    if (tx < 26)
        atomicAdd(&output[tx], SM[tx]);
}



void verify_results(char* input, int* output, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    int temp[26] = {0};
    for(int i=0;i<N;i++) {
        temp[input[i] - 'A']++;
    }
    // Calculate elapsed time
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "CPU Time: " << duration.count() << " ms\n";
    for(int i=0;i<26;i++) {
        // cout << temp[i] << " ";
        assert(temp[i] == output[i]);
    }
}

int main () {
    int N = 1 << 24;
    int bins = 26;
    vector<char> input(N);
    vector<int> result(bins);
    size_t bytes1 = N * sizeof(char);
    size_t bytes2 = 26 * sizeof(int);

    for(int i=0;i<N;i++) {
        input[i] = 65 + rand() % 26;
    }

    // Variable to measure time.
    cudaEvent_t start, stop, kernelStart, kernelStop;
    float time = 0, kernelTime = 0; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    
    cudaEventRecord(start);

    char* d_input;
    int* d_result;
    cudaMalloc(&d_input, bytes1);
    cudaMalloc(&d_result, bytes2);
    cudaMemcpy(d_input, input.data(), bytes1, cudaMemcpyHostToDevice);


    dim3 threads(THREADS);
    // dim3 blocks(N/THREADS > 1 ? N/THREADS : 1);
    dim3 blocks((N / (4*WPT) + THREADS - 1) / THREADS);

	cudaEventRecord(kernelStart);

    histogram_vectorized<<<blocks, threads>>> (d_input, d_result, N);

    // cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
  
    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);

    cudaMemcpy(result.data(), d_result, bytes2, cudaMemcpyDeviceToHost);

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
    // for(auto i:result) cout << i << " ";
    // cout << "\n";
    verify_results(input.data(), result.data(), N);
    cout << "Completed Successfully\n";
    return 0;

}