#ifndef CUDABUFFER_HPP
#define CUDABUFFER_HPP
#include <iostream>
#include <cassert>
#include <cstddef> //for std::size_t
#include "CudaUtils.hpp"

template <typename T>
class CudaBuffer{
public :
  CudaBuffer() = delete;
  explicit CudaBuffer(std::size_t iSize)
      : m_pData(nullptr)
      , m_iSize(iSize)        
  {
    if(m_iSize > 0)
    {
      this->allocate();
    }
    else
    {
      std::cout << "[ERROR]: did not initialize CUDA BUFFER" << std::endl;
      return;
    }
    
  }

  ~CudaBuffer() { cudaFree(m_pData); }
  
  inline T* Data() { return m_pData; }
  inline const T* Data() const { return m_pData; }
  
  void copyFromHost(const T* pData_h);
  void copyFromDevice(T* pData_h);
  
private:
  T* m_pData;
  std::size_t m_iSize;
  void allocate();    
};


template <typename T>
void CudaBuffer<T>::allocate() {
  
  CUDA_CALL(cudaMalloc((void**)&m_pData, m_iSize));

}

template <typename T>
void CudaBuffer<T>::copyFromHost(const T* pData_h) {
  
  CUDA_CALL(cudaMemcpy(m_pData, pData_h, m_iSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaGetLastError());
}



template <typename T>
void CudaBuffer<T>::copyFromDevice(T* pData_h) {

  CUDA_CALL(cudaMemcpy(pData_h, m_pData , m_iSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaGetLastError());

}


#endif 
