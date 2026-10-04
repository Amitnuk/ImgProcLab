#include <iostream>
#include <stdio.h>
#include "imgproc/filters/BoxFilter.hpp"
#include "cuda/CudaBuffer.hpp"

using uchar = unsigned char;

template <class T, class U >
__global__ void BoxFilterKernel(const T* pIn, U* pOut, int iWidth, int iHeight, int iChannels =3, int iKernel=1 )
{
  int col   = threadIdx.x + blockIdx.x * blockDim.x ;
  int row   = threadIdx.y + blockIdx.y * blockDim.y ;
  int depth = threadIdx.z ;

  if( row < iHeight && col < iWidth )
  {
    int iNbPixels = 0;
    int iPixelValue = 0;
    for(int i = -iKernel; i <= iKernel; ++i)
    {
      for(int j = -iKernel; j <= iKernel; ++j)
      {
        int iRow = row + i;
        int iCol = col + j;
        
        if( ( iRow >= 0 && iRow < iHeight ) && ( iCol >= 0 && iCol < iWidth ) )
        {
          int index = ( iRow*iWidth + iCol )*iChannels;
          iPixelValue += pIn[index + depth];
          ++iNbPixels; 
        }
      }
    }
    int index = (row*iWidth + col)*iChannels;
    pOut[index + depth] = static_cast<U>(iPixelValue/iNbPixels);
  }  
}



namespace ImgProc {

  template<typename T, typename U>
  BoxFilter<T,U>::BoxFilter(int iWidth, int iHeight, int iChannels, int iThreads,int iKernelSize)
    : m_iWidth(iWidth)
    , m_iHeight(iHeight)
    , m_iChannels(iChannels)
    , m_iThreads(iThreads)
    , m_iKernelSize(iKernelSize)
  {
    m_iGridDimX = (m_iWidth  + iThreads - 1) / iThreads;
    m_iGridDimY = (m_iHeight + iThreads - 1) / iThreads;
    std::cout << "Average Filter" << std::endl; 
  }
  
  template<typename T, typename U>
  void BoxFilter<T,U>::KernelLauncher(const T* pIn_h, U* pOut_h)
  {
    
    std::cout << "Box Kernel Launcher" << std::endl;
    dim3 oGridDim(m_iGridDimX, m_iGridDimY,1);
    dim3 oBlockDim(m_iThreads, m_iThreads, m_iChannels);
    
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


  template class BoxFilter<uchar, uchar>;


}
