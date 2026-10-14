#include <iostream>
#include <stdio.h>
#include "ImgProc/filters/MedianFilter.hpp"
#include "cuda/CudaBuffer.hpp"

using uchar = unsigned char;


template <class T>
__device__ void sort(T* pIn, int N)
{
  for (size_t i = 0; i < N - 1; i++)
  {
    for (size_t j = i+1; j < N; j++)
    {
  
        // naive approach to avoid divergence
        const T a = pIn[i];
        const T b = pIn[j] ;

        bool bSwap = (a > b); 

        pIn[i] = bSwap ? b : a;
        pIn[j] = bSwap ? a : b;
    }
    
  }
  
}



template <class T, class U >
__global__ void MedianFilterKernel(const T* pIn, U* pOut, int iWidth, int iHeight, int iChannels = 3, int iKernel=3 )
{
  int col   = threadIdx.x + blockIdx.x * blockDim.x ;
  int row   = threadIdx.y + blockIdx.y * blockDim.y ;
 
  extern __shared__ T windows[];

  if( row < iHeight && col < iWidth )
  {
    int iHalf = iKernel/2;
    int iWindowSize = iKernel*iKernel;

    int indice = threadIdx.y*blockDim.x + threadIdx.x;
    
    T* pWindowB = &windows[indice*iWindowSize*iChannels];
    T* pWindowG = pWindowB + iWindowSize ;
    T* pWindowR = pWindowG + iWindowSize ;
    for(int i = -iHalf; i <= iHalf; ++i)
    {
      for(int j = -iHalf; j <= iHalf; ++j)
      {

        // clamping the border allows us to avoid divergence when checking outbound pixels
        int r = max(0, min(row + i, iHeight - 1));
        int c = max(0, min(col + j, iWidth - 1)); 

        int windowRow = i + iHalf;
        int windowCol = j + iHalf;

        int iWindowIndex = windowRow*iKernel + windowCol;

        int iImageIndex = ( r*iWidth + c );
        pWindowB[iWindowIndex] = pIn[iImageIndex*iChannels + 0];
        pWindowG[iWindowIndex] = pIn[iImageIndex*iChannels + 1];
        pWindowR[iWindowIndex] = pIn[iImageIndex*iChannels + 2];
        
      }
    }
    sort<T>(pWindowB, iWindowSize);
    pOut[(row * iWidth + col) * iChannels + 0] = static_cast<U>(pWindowB[iWindowSize/2]);
    sort<T>(pWindowG, iWindowSize);
    pOut[(row * iWidth + col) * iChannels + 1] = static_cast<U>(pWindowG[iWindowSize/2]);
    sort<T>(pWindowR, iWindowSize);
    pOut[(row * iWidth + col) * iChannels + 2] = static_cast<U>(pWindowR[iWindowSize/2]);
  }  
}


namespace ImgProc {

  template<typename T, typename U>
  Filter::MedianFilter<T,U>::MedianFilter(int iWidth, int iHeight, int iChannels, int iThreads,int iKernelSize)
    : m_iWidth(iWidth)
    , m_iHeight(iHeight)
    , m_iChannels(iChannels)
    , m_iThreads(iThreads)
    , m_iKernelSize(iKernelSize)
  {
    
    std::cout << "Median Filter" << std::endl; 

    if(iKernelSize > 7)
    {
      std::cout << "[WARNING] : Max Kernel Size Should be 7" << std::endl;
    }

    if( iKernelSize%2 == 0)
    {
      std::cout << "[WARNING] : A KERNEL SIZE SHOULD BE A ODD NUMBER SUP THAN 1, eg 3, 5, 7, 9, ..., 2*n+1" << std::endl;
    }

    if( m_iKernelSize < 3 || m_iKernelSize > 7|| m_iKernelSize % 2 == 0 )
    {
      throw std::invalid_argument("m_iKernelSize must be odd : [3 5 or 7]");
    }
    
    m_iGridDimX = (m_iWidth  + m_iThreads - 1) / m_iThreads;
    m_iGridDimY = (m_iHeight + m_iThreads - 1) / m_iThreads;
    std::cout << "( " << m_iGridDimX << " , " << m_iGridDimY << " , " << m_iThreads << " , " << m_iKernelSize << " )"<< std::endl;
  }
  
  template<typename T, typename U>
  void Filter::MedianFilter<T,U>::KernelLauncher(const T* pIn_h, U* pOut_h)
  {
    
    std::cout << "Median Kernel Launcher" << std::endl;
    dim3 oGridDim(m_iGridDimX, m_iGridDimY);
    dim3 oBlockDim(m_iThreads, m_iThreads);
    
    int iSize = m_iHeight*m_iWidth*m_iChannels * sizeof(U);

    CudaBuffer<T> oCudaBufferIn(iSize);
    CudaBuffer<U> oCudaBufferOut(iSize);

    oCudaBufferIn.copyFromHost(pIn_h);

    size_t sharedBytes = oBlockDim.x * oBlockDim.y * m_iKernelSize * m_iKernelSize * m_iChannels * sizeof(T);

    cudaEvent_t oStart, oStop;
    float fTimeInMS;

    CUDA_CHECK(cudaEventCreate(&oStart));
    CUDA_CHECK(cudaEventCreate(&oStop));
    CUDA_CHECK(cudaEventRecord(oStart));

    MedianFilterKernel<T,U><<<oGridDim, oBlockDim, sharedBytes>>>(oCudaBufferIn.Data(), oCudaBufferOut.Data(), m_iWidth, m_iHeight, m_iChannels, m_iKernelSize);

    CUDA_CHECK(cudaEventRecord(oStop));
    CUDA_CHECK(cudaEventSynchronize(oStop));
  
    CUDA_CHECK(cudaEventElapsedTime(&fTimeInMS, oStart, oStop));
    std::cout << "CUDA kernel: " << fTimeInMS << "ms\n";

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

  
    oCudaBufferOut.copyFromDevice(pOut_h);
  }


  template class Filter::MedianFilter<uchar, uchar>;
}
