#pragma once 
#include <cuda_runtime.h>
#include "cuda/common/CudaBuffer.hpp"


using uchar = unsigned char;
struct CudaExecutionContext
{
    CudaBuffer<uchar> m_oDeviceInput;
    CudaBuffer<float> m_oDeviceIntermediate;
    CudaBuffer<uchar> m_oDeviceOutputs;   
    CudaBuffer<float> m_oKernel;

    std::size_t m_iWidth;
    std::size_t m_iHeight;
    
    cudaStream_t m_oStream = nullptr;
};
