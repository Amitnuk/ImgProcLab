#pragma once
#include "CudaUtils.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept>


template<typename LaunchFunction>
float executeCuda(LaunchFunction&& launch, const bool& bBenchmark = false, const int& iWarmups = 10, const int& iIterations = 100)
{
    float fElapsedTime = 0.0f;
#ifdef PROFILE_CUDA
        if(!bBenchmark)
        {
            launch();        
        }
        else
        {
            throw std::invalid_argument("Benchmarking is unavailable in PROFILE_CUDA mode");
        }
#else

#ifdef IMGPROCLAB_DEBUG
    if(bBenchmark)
        std::cerr
            << "[WARNING] Benchmarking in Debug mode.\n"
            << "CUDA -G disables many optimizations.\n"
            << "Results are not representative of Release performance.\n"
            << "Add -DCMAKE_BUILD_TYPE=Release instead\n";
#endif

    if(!bBenchmark)
    {
        launch();        
    }
    else
    {

        for(int i = 0; i < iWarmups; i++)
        {
            launch();
        }

        CUDA_CHECK(cudaGetLastError());

        cudaEvent_t oStart, oStop;
       
        CUDA_CHECK(cudaEventCreate(&oStart));
        CUDA_CHECK(cudaEventCreate(&oStop));
        CUDA_CHECK(cudaEventRecord(oStart));

        for (int i = 0; i < iIterations; i++)
        {
            launch();
        }
        
        CUDA_CHECK(cudaEventRecord(oStop))
        CUDA_CHECK(cudaEventSynchronize(oStop));
        CUDA_CHECK(cudaEventElapsedTime(&fElapsedTime, oStart, oStop));
        fElapsedTime /= static_cast<float>(iIterations);
        std::cout << "Average kernel time: "<< fElapsedTime << "ms\n";
    }
#endif
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return fElapsedTime;
}