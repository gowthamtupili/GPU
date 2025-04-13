#include <iostream>
#include <vector>
#include <cassert>
#include <cuda.h>
#include <chrono>

 
using std::cout;
using std::vector;

#define THREADS 256
#define bins 26

const char *ptx_source = R"(
    .version 8.5
    .target sm_52
    .address_size 64


    .visible .entry _histotiled(
        .param .u64 _histotiled_param_0,
        .param .u64 _histotiled_param_1,
        .param .u32 _histotiled_param_2
    )
    {
        .reg .pred 	%p<4>;
        .reg .b32 %tx, %tide, %n, %smInd, %smBase, %smVal, %outputArrVal,%r3,%r4,%r5,%r6,%r9,%r10,%r12,%r14,%r15,%r20;
        .reg .b64 %input_ptr, %TideX, %inputVal, %output_ptr, %rd1, %rd2, %rd8, %rd9, %rd10 ;
        .reg .b16 %inputArrVal;

        .shared .align 4 .b32 SM[26];

        ld.param.u64 %rd1, [_histotiled_param_0];
        ld.param.u64 %rd2, [_histotiled_param_1];
        ld.param.u32 %n, [_histotiled_param_2];


        mov.u32 %r3, %ctaid.x;
        mov.u32 %r4, %ntid.x;
        mov.u32 %tx, %tid.x; // r5 -> threadidx.x (tx)
        mad.lo.s32 %tide, %r4, %r3, %tx; // tid calc (r1 = tid(global thread id))


        setp.ge.s32	%p1, %tide, %n; // if statement (tid < N)
        // @%p1 bra $exit;

        setp.gt.s32 %p2, %tx, 25; // if(tx < 26) case
        // // mul.wide.u32 %r6, %r5, 4;
        // // Is problem from mul calc ???
        shl.b32 %smInd, %tx, 2;
        mov.u32 %smBase, SM;
        add.u32 %smVal, %smBase, %smInd; 
        @%p2 bra $check1;
        mov.u32 	%r20, 0;
        st.shared.u32 [%smVal], %r20; // Store 0 in tx < 26;
    $check1:
        bar.sync 0;
        @%p1 bra 	$check2;
        cvt.s64.s32 %TideX, %tide;
        cvta.to.global.u64 	%input_ptr, %rd1;
        add.s64 %inputVal, %input_ptr, %TideX;
        ld.global.s8 	%inputArrVal, [%inputVal];
        mul.wide.s16 	%r10, %inputArrVal, 4;
        add.s32 	%r12, %smBase, %r10;
        add.s32 	%outputArrVal, %r12, -260;
        atom.shared.add.u32 	%r14, [%outputArrVal], 1;

    $check2:
        bar.sync 0;
        @%p2 bra $exit;
        cvta.to.global.u64 	%output_ptr, %rd2;
        mul.wide.s32 %rd10, %tx, 4;
        // cvt.s64.s32 %rd10, %smInd;
        add.s64 %rd8, %output_ptr, %rd10;
        ld.shared.u32 %r9, [%smVal];
        atom.global.add.u32 %r15, [%rd8], %r9;

    $exit:
        ret;

    }
)";

void checkCudaError(CUresult result, const char* funcName) {
    if (result != CUDA_SUCCESS) {
        const char* errorStr;
        cuGetErrorString(result, &errorStr);
        std::cerr << "CUDA Driver API Error in " << funcName << ": " << errorStr << std::endl;
        exit(EXIT_FAILURE);
    }
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
        // cout << output[i] << " " << temp[i] << "\n";
        assert(temp[i] == output[i]);
    }
}

int main () {
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;
    CUdeviceptr d_input, d_output;

    int N = 1 << 24;
    // int N = 10;
    // int bins = 26;
    vector<char> input(N);
    vector<int> result(bins, 0);
    size_t bytes1 = N * sizeof(char);
    size_t bytes2 = bins * sizeof(int);

    for(int i=0;i<N;i++) {
        input[i] = 65 + rand() % 26;
    }

        
    // Initialize CUDA Driver API
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);

    // Load PTX
    cuModuleLoadData(&cuModule, ptx_source);
    cuModuleGetFunction(&cuFunction, cuModule, "_histotiled");

    CUresult err;

    CUevent start, stop, kernelStart, kernelStop;
    float time = 0, kernelTime = 0; 
    cuEventCreate(&start, CU_EVENT_DEFAULT);
    cuEventCreate(&stop, CU_EVENT_DEFAULT);
    cuEventCreate(&kernelStart, CU_EVENT_DEFAULT);
    cuEventCreate(&kernelStop, CU_EVENT_DEFAULT);


    cuEventRecord(start, 0);

    err = cuMemAlloc(&d_input, bytes1);
    checkCudaError(err, "cuMemAlloc");
    cuMemAlloc(&d_output, bytes2);
    err = cuMemcpyHtoD(d_input, input.data(), bytes1);
    checkCudaError(err, "cuMemcpyHtoD");
    // cuMemcpyHtoD(d_b, b.data(), size);

    void *args[] = { &d_input, &d_output, &N };

    cuEventRecord(kernelStart, 0);

    
    err = cuLaunchKernel(cuFunction, N / THREADS, 1, 1, THREADS, 1, 1, 0, 0, args, 0);
    checkCudaError(err, "cuLaunchKernel");

    cuEventRecord(kernelStop, 0);
    cuEventSynchronize(kernelStop);

    err = cuMemcpyDtoH(result.data(), d_output, bytes2);
    checkCudaError(err, "cuMemcpyDtoH");

    cuEventRecord(stop, 0);
    cuEventSynchronize(stop);

    cuEventElapsedTime(&time, start, stop);
    cuEventElapsedTime(&kernelTime, kernelStart, kernelStop);
    std::cout << "GPU Kernel time: " << kernelTime << " ms" << std::endl;
    std::cout << "GPU Total time (H2D + Kernel + D2H): " << time << " ms" << std::endl;

    cuMemFree(d_input); cuMemFree(d_output);
    cuCtxDestroy(cuContext);

    verify_results(input.data(), result.data(), N);
    cout << "Completed Successfully\n";
    return 0;
 
}