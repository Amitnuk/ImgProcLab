#include <iostream>
#include <stdio.h>
#include "ImgProc/filters/BoxFilter.hpp"
#include "cuda/CudaBuffer.hpp"

using uchar = unsigned char;

template <class T, class U >
__global__ void BoxFilterKernel(const T* pIn, U* pOut, int iWidth, int iHeight, int iChannels =3, const std::size_t iKernelSize=3 )
{
  int col   = threadIdx.x + blockIdx.x * blockDim.x ;
  int row   = threadIdx.y + blockIdx.y * blockDim.y ;
  
  if( row < iHeight && col < iWidth )
  {
    int iNbPixels = 0;
    int vBGR[3] = {0, 0, 0}; 
    int iHalf = iKernelSize/2;
    for(int i = -iHalf; i <= iHalf; ++i)
    {
      for(int j = -iHalf; j <= iHalf; ++j)
      {
        int iRow = row + i + iHalf;
        int iCol = col + j + iHalf;
        
        if( ( iRow >= 0 && iRow < iHeight ) && ( iCol >= 0 && iCol < iWidth ) )
        {
          int index = ( iRow*iWidth + iCol )*iChannels;
          vBGR[0] += pIn[index];
          vBGR[1] += pIn[index + 1];
          vBGR[2] += pIn[index + 2];
          ++iNbPixels; 
        }
      }
    }
    int index = (row*iWidth + col)*iChannels;
    pOut[index] = static_cast<U>(vBGR[0]/iNbPixels);
    pOut[index+1] = static_cast<U>(vBGR[1]/iNbPixels);
    pOut[index+2] = static_cast<U>(vBGR[2]/iNbPixels);
  }  
}


namespace ImgProc {

  template<typename T, typename U>
  Filter::BoxFilter<T,U>::BoxFilter(int iWidth, int iHeight, int iChannels, int iThreads,int iKernelSize)
    : m_iWidth(iWidth)
    , m_iHeight(iHeight)
    , m_iChannels(iChannels)
    , m_iThreads(iThreads)
    , m_iKernelSize(iKernelSize)
  {

    std::cout << "Box Filter" << std::endl; 
    if( m_iKernelSize%2 == 0)
    {
      std::cout << "[WARNING] : A KERNEL SIZE SHOULD BE A ODD NUMBER SUP THAN 1, eg 3, 5, 7, 9, ..., 2*n+1" << std::endl;
    }

    if( m_iKernelSize < 3 || m_iKernelSize % 2 == 0 )
    {
      throw std::invalid_argument("m_iKernelSize must be odd and >= 3");
    }
          
    m_iGridDimX = (m_iWidth  + m_iThreads - 1) / m_iThreads;
    m_iGridDimY = (m_iHeight + m_iThreads - 1) / m_iThreads;
    std::cout << "( " << m_iGridDimX << " , " << m_iGridDimY << " , " << m_iThreads << " , " << m_iKernelSize << " )"<< std::endl;
  }
  
  template<typename T, typename U>
  void Filter::BoxFilter<T,U>::KernelLauncher(const T* pIn_h, U* pOut_h)
  {
    
    std::cout << "Box Kernel Launcher" << std::endl;
    dim3 oGridDim(m_iGridDimX, m_iGridDimY);
    dim3 oBlockDim(m_iThreads, m_iThreads);
    
    int iSize = m_iHeight*m_iWidth*m_iChannels * sizeof(U);

    CudaBuffer<T> oCudaBufferIn(iSize);
    CudaBuffer<U> oCudaBufferOut(iSize);

    oCudaBufferIn.copyFromHost(pIn_h);

    cudaEvent_t oStart, oStop;
    float fTimeInMS;

    CUDA_CALL(cudaEventCreate(&oStart));
    CUDA_CALL(cudaEventCreate(&oStop));
    CUDA_CALL(cudaEventRecord(oStart));

    BoxFilterKernel<T,U><<<oGridDim, oBlockDim>>>(oCudaBufferIn.Data(), oCudaBufferOut.Data(), m_iWidth, m_iHeight, m_iChannels, m_iKernelSize);

    CUDA_CALL(cudaEventRecord(oStop));
    CUDA_CALL(cudaEventSynchronize(oStop));
  
    CUDA_CALL(cudaEventElapsedTime(&fTimeInMS, oStart, oStop));
    std::cout << "CUDA kernel: " << fTimeInMS << "ms\n";

    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

  
    oCudaBufferOut.copyFromDevice(pOut_h);
  }


  template class Filter::BoxFilter<uchar, uchar>;
}
