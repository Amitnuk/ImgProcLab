#pragma once
#include "CudaUtils.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept> 
#include <chrono>

enum class Backend
{   
    CUDA,
    CPU
};

template<typename LaunchFunction>
float executeOpenCV(LaunchFunction&&  launch, const Backend& backend, const bool& bBenchmark, const int& iWarmups = 10, const int& iIterations = 100)
{
    if(!bBenchmark)
    {
        throw std::invalid_argument("Can only launch comparision while benchmarking, add --benchmark");
    }

    float fTimeElapsed = 0.0f;


#ifdef PROFILE_CUDA
    throw std::invalid_argument("Can not launch comparision while cuda profiling is activated, add -DPROFILE_CUDA=OFF");
#endif

    if(backend == Backend::CPU)
    {
        auto oStart = std::chrono::steady_clock::now();

        for(int i = 0; i< iWarmups; i++)
        {
            launch();
        }

        auto oStop = std::chrono::steady_clock::now();

        fTimeElapsed = std::chrono::duration<float,std::milli>(oStop - oStart).count();
        fTimeElapsed /= static_cast<float>(iIterations);
        std::cout << "Average OpenCV CPU Kernel time: "<< fTimeElapsed << "ms\n";
    }
    else if(backend == Backend::CUDA)
    {
        for(int i = 0; i< iWarmups; i++)
        {
            launch();
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEvent_t oStart, oStop;
        CUDA_CHECK(cudaEventCreate(&oStart));
        CUDA_CHECK(cudaEventCreate(&oStop)); 
        CUDA_CHECK(cudaEventRecord(oStart)); 

        for(int i = 0; i < iIterations; i++)
        {
            launch();
        }

        CUDA_CHECK(cudaEventRecord(oStop));
        CUDA_CHECK(cudaEventSynchronize(oStop));
        CUDA_CHECK(cudaEventElapsedTime(&fTimeElapsed, oStart, oStop));
        
        fTimeElapsed /= static_cast<float>(iIterations);
        std::cout << "Average OpenCV CUDA Kernel time: "<< fTimeElapsed << "ms\n";
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else
    {
        throw std::invalid_argument("Unkown mode, only two modes available GPU or CUDA");
    }
    

    return fTimeElapsed;
}