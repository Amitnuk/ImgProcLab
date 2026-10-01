#ifndef CUDABUFFER_HPP
#define CUDABUFFER_HPP
#include "CudaUtils.hpp"
#include <cassert>
#include <cstddef> //for std::size_t

template <typename T>
class CudaBuffer{
public :
  explicit CudaBuffer(std::size_t iSize)
      : m_pData(nullptr)
      , m_iSize(iSize)        
  {
    this->allocate();
  }

  ~CudaBuffer() { cudaFree(m_pData); }
  
  inline T* Data() { return m_pData; }
  inline const T* Data() const { return m_pData; }
  
  void copyFromHost(const T* pData_h,  std::size_t iSize);
  void copyFromDevice(T* pData_h, std::size_t iSize);
  
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
void CudaBuffer<T>::copyFromHost(const T* pData_h, std::size_t iSize) {
  assert(iSize == m_iSize);
  if (iSize != m_iSize) {
    std::cout << "[WARNING]: Array Size MisMatch" << std::endl;
    return ;
  }
  
  CUDA_CALL(cudaMemcpy(m_pData, pData_h, iSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaGetLastError());
}



template <typename T>
void CudaBuffer<T>::copyFromDevice(T* pData_h, std::size_t iSize) {

  assert(iSize == m_iSize);
  if (iSize != m_iSize) {
    std::cout << "[WARNING]: Array Size MisMatch" << std::endl;
    return ;
  }
  
  CUDA_CALL(cudaMemcpy(pData_h, m_pData , iSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaGetLastError());

}


#endif 
