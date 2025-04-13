// Naive Implementation

#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>


using std::cout;
using std::vector;


#define THREADS 256

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
  


__global__ void histogram(char* input, int* output, int N) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < N) {
        atomicAdd(&output[input[tid] - 'A'], 1);
    }
};



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
    dim3 blocks(N/THREADS);

	cudaEventRecord(kernelStart);

    histogram<<<blocks, threads>>> (d_input, d_result, N);

    cudaDeviceSynchronize();
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
