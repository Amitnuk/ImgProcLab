#include <iostream>
#include <stdio.h>
#include "ImgProc/colorConversion/BGR2Gray.hpp"
#include "cuda/CudaBuffer.hpp"

using uchar = unsigned char;

template <class T, class U >
__global__ void BGR2GrayKernel( const U* Pin, T* Pout,  int width, int height, const int iChannels = 3) 
{
    int col = threadIdx.x + blockIdx.x * blockDim.x ;
    int row = threadIdx.y + blockIdx.y * blockDim.y ;

    if ( col < width &&  row < height )
    {
      int grayOffSet = row*width + col;
      int rgbOffSet = grayOffSet * iChannels ;
      U b = Pin[rgbOffSet] ;
      U g = Pin[rgbOffSet + 1] ;
      U r = Pin[rgbOffSet + 2] ;
      Pout[grayOffSet] = static_cast<T>(0.21f*r + 0.71f*g + 0.07f*b);
      
    }
    

}



namespace ImgProc {

  template<typename T, typename U>
  color::BGR2Gray<T,U>::BGR2Gray(int iWidth, int iHeight, int iChannels, int iThreads)
    : m_iWidth(iWidth)
    , m_iHeight(iHeight)
    , m_iChannels(iChannels)
    , m_iThreads(iThreads)
  {

    std::cout << "Color Conversion" << std::endl; 
    m_iGridDimX = (m_iWidth  + iThreads - 1) / iThreads;
    m_iGridDimY = (m_iHeight + iThreads - 1) / iThreads;
  }
  
  template<typename T, typename U>
  void color::BGR2Gray<T,U>::KernelLauncher(const T* pIn_h, U* pOut_h)
  {
    
    std::cout << "Color Conversion Launcher" << std::endl;
    dim3 oGridDim(m_iGridDimX, m_iGridDimY,1);
    dim3 oBlockDim(m_iThreads, m_iThreads, 1);
    
    int iSizeIn = m_iHeight*m_iWidth*m_iChannels * sizeof(U);
    int iSizeOut = m_iHeight*m_iWidth * sizeof(U);

    CudaBuffer<T> oCudaBufferIn(iSizeIn);
    CudaBuffer<U> oCudaBufferOut(iSizeOut);

    oCudaBufferIn.copyFromHost(pIn_h);

    cudaEvent_t oStart, oStop;
    float fTimeInMS;

    CUDA_CALL(cudaEventCreate(&oStart));
    CUDA_CALL(cudaEventCreate(&oStop));
    CUDA_CALL(cudaEventRecord(oStart));

    BGR2GrayKernel<T,U><<<oGridDim, oBlockDim>>>(oCudaBufferIn.Data(), oCudaBufferOut.Data(), m_iWidth, m_iHeight, m_iChannels);

    CUDA_CALL(cudaEventRecord(oStop));
    CUDA_CALL(cudaEventSynchronize(oStop));
  
    CUDA_CALL(cudaEventElapsedTime(&fTimeInMS, oStart, oStop));
    std::cout << "CUDA kernel: " << fTimeInMS << "ms\n";

    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

  
    oCudaBufferOut.copyFromDevice(pOut_h);
  }


  template class color::BGR2Gray<uchar, uchar>;
}
