#include <iostream>
#include <stdio.h>
#include "imgproc/filters/BoxFilter.hpp"



#define CUDA_CALL( call )         \ 
{                                  \
  cudaError_t err = call;           \
  if( err != cudaSuccess )          \ 
    std::cerr << "Cuda Error " << err << "in " << __FILE__ <<":"<<__LINE__ << " : " << cudaGetErrorString(err) << " ( "<< #call << " ) " << std::endl ;\
} \



using uchar=unsigned char;
__global__ void AverageFilterKernel(const uchar* pIn, uchar* pOut, int iWidth, int iHeight, int iChannels =3, int iKernel=1 )
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
    pOut[index + depth] = static_cast<uchar>(iPixelValue/iNbPixels);
  }  
}



namespace ImgProc {


  BoxFilter::BoxFilter(int iWidth, int iHeight, int iChannels, int iThreads,int iKernelSize)
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
  
  void BoxFilter::KernelLauncher(const uchar* pIn_h,uchar* pOut_h)
  {
    
    std::cout << "Kernel Launcher" << std::endl;
    dim3 oGridDim(m_iGridDimX, m_iGridDimY,1);
    dim3 oBlockDim(m_iThreads, m_iThreads, m_iChannels);
    
    uchar* pIn_d;
    uchar* pOut_d;
    int iSize = m_iHeight*m_iWidth*m_iChannels * sizeof(uchar);
    
    CUDA_CALL(cudaMalloc((void**)&pOut_d, iSize ));
    CUDA_CALL(cudaMalloc((void**)&pIn_d, iSize ));
    CUDA_CALL(cudaMemcpy(pIn_d, pIn_h, iSize, cudaMemcpyHostToDevice));
    


    cudaEvent_t oStart, oStop;


    CUDA_CALL(cudaEventCreate(&oStart));
    CUDA_CALL(cudaEventCreate(&oStop));
    CUDA_CALL(cudaEventRecord(oStart));

    AverageFilterKernel<<<oGridDim, oBlockDim>>>(pIn_d, pOut_d, m_iWidth, m_iHeight, m_iChannels, m_iKernelSize);
  
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());


    CUDA_CALL(cudaEventRecord(oStop));
    CUDA_CALL(cudaEventSynchronize(oStop));

    float fTimeInMS;
    CUDA_CALL(cudaEventElapsedTime(&fTimeInMS, oStart, oStop));

   
    

    std::cout << "CUDA kernel: " << fTimeInMS << " ms\n";

    CUDA_CALL(cudaMemcpy(pOut_h, pOut_d, iSize, cudaMemcpyDeviceToHost));

    

    CUDA_CALL(cudaFree(pIn_d));
    CUDA_CALL(cudaFree(pOut_d));
  }
}
