#include <iostream>
#include <stdio.h>
#include "imgproc/filters/segmentation/BinaryThreshold.hpp"
#include "cuda/CudaBuffer.hpp"



using uchar = unsigned char;


template<typename T, typename U >
__global__ void BinaryThresholdKernel(const T* pIn, U* pOut, int iHeight, int iWidth, int iThreshold)
{
  int Col = threadIdx.x + blockIdx.x * blockDim.x ;
  int Row = threadIdx.y + blockIdx.y * blockDim.y ;
  
  if(Col < iWidth && Row < iHeight)
  {
    int index = Row * iWidth + Col;
    pOut[index] = (pIn[index] > iThreshold) ? 255 : 0 ;
  }
}

namespace ImgProc{

  namespace Seg
  {
    template <typename T, typename U>
    BinaryThreshold<T, U>::BinaryThreshold(std::size_t iWidth /*= 1*/, std::size_t iHeight /*= 32*/,int iThreads /*= 16*/, int iThreshold /*=150*/)
      
      : m_iWidth(iWidth)
      , m_iHeight(iHeight)
      , m_iThreads(iThreads)
      , m_iThreshold(iThreshold)
    {
      m_iGridDimX = (m_iWidth  + m_iThreads - 1)/m_iThreads;
      m_iGridDimY = (m_iHeight + m_iThreads - 1)/m_iThreads;
    }

    template <typename T, typename U>
    void BinaryThreshold<T, U>::KernelLauncher(const T *pIn_h, U *pOut_h)
    {
      std::cout << "Binary Threshold Kernel Launcher " << std::endl;
      dim3 oGridDim(m_iGridDimX, m_iGridDimY);
      dim3 oBlockDim(m_iThreads, m_iThreads); 

      int iImageSize = m_iWidth * m_iHeight * sizeof(T);
      CudaBuffer<T> oCudaBufferIn(iImageSize);
      CudaBuffer<T> oCudaBufferOut(iImageSize);

      oCudaBufferIn.copyFromHost(pIn_h);

      cudaEvent_t oStart, oStop;
      float fTimeInMS;

      CUDA_CALL(cudaEventCreate(&oStart));
      CUDA_CALL(cudaEventCreate(&oStop));
      CUDA_CALL(cudaEventRecord(oStart));

      BinaryThresholdKernel<T, U><<<oGridDim,oBlockDim>>>(oCudaBufferIn.Data(), oCudaBufferOut.Data(), m_iHeight, m_iWidth, m_iThreshold);

      CUDA_CALL(cudaEventRecord(oStop));
      CUDA_CALL(cudaEventSynchronize(oStop));
    
      CUDA_CALL(cudaEventElapsedTime(&fTimeInMS, oStart, oStop));
      std::cout << "CUDA kernel: " << fTimeInMS << "ms\n";

      CUDA_CALL(cudaGetLastError());
      CUDA_CALL(cudaDeviceSynchronize());

    
      oCudaBufferOut.copyFromDevice(pOut_h);
    }    

    template class BinaryThreshold<int, int>;
    template class BinaryThreshold<uchar, uchar>;
  } 
}
