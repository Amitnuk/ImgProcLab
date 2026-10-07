#include <iostream>
#include <stdio.h>
#include "imgproc/filters/blur/MedianFilter.hpp"
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
__global__ void MedianFilterKernel(const T* pIn, U* pOut, int iWidth, int iHeight, int iChannels =3, int iKernel=1 )
{
  int col   = threadIdx.x + blockIdx.x * blockDim.x ;
  int row   = threadIdx.y + blockIdx.y * blockDim.y ;
  int depth = threadIdx.z ;
  extern __shared__ T windows[];

  if( row < iHeight && col < iWidth )
  {
    int iHalf = iKernel/2;
    int iWindonSize = iKernel*iKernel;

    int indice = threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x + threadIdx.x;

    T* pWindow = &windows[indice*iWindonSize];
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
        pWindow[iWindowIndex] = pIn[iImageIndex*iChannels + depth];
        
      }
    }
    sort<T>(pWindow, iWindonSize);
    T evenPixel = pWindow[(iWindonSize-1)/2] ;
    T oddPixel = pWindow[iWindonSize/2];
    bool bOdd = (iKernel%2 != 0);
    T tPixelValue = bOdd ? oddPixel : ( evenPixel + oddPixel )/2;
    pOut[(row * iWidth + col) * iChannels + depth] = static_cast<U>(tPixelValue);
  }  
}


namespace ImgProc {

  template<typename T, typename U>
  Blur::MedianFilter<T,U>::MedianFilter(int iWidth, int iHeight, int iChannels, int iThreads,int iKernelSize)
    : m_iWidth(iWidth)
    , m_iHeight(iHeight)
    , m_iChannels(iChannels)
    , m_iThreads(iThreads)
    , m_iKernelSize(iKernelSize)
  {
    m_iGridDimX = (m_iWidth  + iThreads - 1) / iThreads;
    m_iGridDimY = (m_iHeight + iThreads - 1) / iThreads;
    std::cout << "Median Filter" << std::endl; 
    if(iKernelSize > 7)
    {
      std::cout << "[WARNING] : Max Kernel Size Should be 7" << std::endl;
    }
    
  }
  
  template<typename T, typename U>
  void Blur::MedianFilter<T,U>::KernelLauncher(const T* pIn_h, U* pOut_h)
  {
    
    std::cout << "Median Kernel Launcher" << std::endl;
    dim3 oGridDim(m_iGridDimX, m_iGridDimY,1);
    dim3 oBlockDim(m_iThreads, m_iThreads, m_iChannels);
    
    int iSize = m_iHeight*m_iWidth*m_iChannels * sizeof(U);

    CudaBuffer<T> oCudaBufferIn(iSize);
    CudaBuffer<U> oCudaBufferOut(iSize);

    oCudaBufferIn.copyFromHost(pIn_h);

    size_t sharedBytes = oBlockDim.x * oBlockDim.y * oBlockDim.z * m_iKernelSize * m_iKernelSize * sizeof(T);

    cudaEvent_t oStart, oStop;
    float fTimeInMS;

    CUDA_CALL(cudaEventCreate(&oStart));
    CUDA_CALL(cudaEventCreate(&oStop));
    CUDA_CALL(cudaEventRecord(oStart));

    MedianFilterKernel<T,U><<<oGridDim, oBlockDim, sharedBytes>>>(oCudaBufferIn.Data(), oCudaBufferOut.Data(), m_iWidth, m_iHeight, m_iChannels, m_iKernelSize);

    CUDA_CALL(cudaEventRecord(oStop));
    CUDA_CALL(cudaEventSynchronize(oStop));
  
    CUDA_CALL(cudaEventElapsedTime(&fTimeInMS, oStart, oStop));
    std::cout << "CUDA kernel: " << fTimeInMS << "ms\n";

    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

  
    oCudaBufferOut.copyFromDevice(pOut_h);
  }


  template class Blur::MedianFilter<uchar, uchar>;
}
