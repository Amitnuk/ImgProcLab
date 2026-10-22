#ifndef CUDABUFFER_HPP
#define CUDABUFFER_HPP
#include <iostream>
#include <cassert>
#include <cstddef> //for std::size_t
#include "CudaUtils.hpp"
#include <stdexcept>

//TODO: Must handle cuda funtion failing
template <typename T>
class CudaBuffer{
public :
  CudaBuffer()
  :m_pData(nullptr)
  ,m_iSize(0)        
  {

  }

  explicit CudaBuffer(std::size_t iSize)
      : m_pData(nullptr)
      , m_iSize(iSize)        
  {
    if(m_iSize > 0)
    {
      this->_allocate();
    }
    else
    {
      throw std::invalid_argument("CudaBuffer size must be greater than zero");
    }
    
  }

  CudaBuffer(const CudaBuffer<T>& oCudaBuffer)
  {
    m_iSize = oCudaBuffer.m_iSize;
    CUDA_CHECK(cudaMalloc((void**)&m_pData, oCudaBuffer.m_iSize));
    CUDA_CHECK(cudaMemcpy(m_pData, oCudaBuffer.m_pData, oCudaBuffer.m_iSize, cudaMemcpyDeviceToDevice ));
  }

  CudaBuffer<T>& operator=(const CudaBuffer<T> &oCudaBuffer)
  {

    if(&oCudaBuffer == this)
    {
      return *this;
    }

    CUDA_CHECK(cudaFree(m_pData));
    m_iSize = oCudaBuffer.m_iSize;
    CUDA_CHECK(cudaMalloc((void**)&m_pData, oCudaBuffer.m_iSize));
    CUDA_CHECK(cudaMemcpy(m_pData, oCudaBuffer.m_pData, oCudaBuffer.m_iSize, cudaMemcpyDeviceToDevice ));

    return *this;
  }

  CudaBuffer(CudaBuffer<T>&& oCudaBuffer) noexcept  
  : m_pData(oCudaBuffer.m_pData)   
  , m_iSize(oCudaBuffer.m_iSize)
  {
    oCudaBuffer.m_pData = nullptr;
    oCudaBuffer.m_iSize = 0;
  }

  CudaBuffer<T>& operator=(CudaBuffer<T> &&oCudaBuffer) noexcept
  {

    if(&oCudaBuffer == this)
    {
      return *this;
    }

    CUDA_CHECK(cudaFree(m_pData));
    m_iSize = oCudaBuffer.m_iSize;
    m_pData = oCudaBuffer.m_pData;
    
    oCudaBuffer.m_pData = nullptr;
    oCudaBuffer.m_iSize = 0;

    return *this;
    
  }    

  ~CudaBuffer() { CUDA_CHECK(cudaFree(m_pData)); m_iSize = 0; }
  
  inline T* Data() { return m_pData; }
  inline const T* Data() const { return m_pData; }

  T* operator->() {return m_pData;}
  const T* operator->() const {return m_pData;}

  operator T*() noexcept { return m_pData;}
  operator const T*() const noexcept {return m_pData;}

  const T& operator*() const {return *m_pData;}
  T& operator*() {return *m_pData;}
  

  void copyFromHost(const T* pData_h);
  void copyFromDevice(T* pData_h) const;
  void allocate(std::size_t iSize) 
  {  
    if(m_iSize != 0)
    {
      throw std::invalid_argument("Changing a size of existing CudaBuffer is not allowed");
    }
    
    if(iSize <= 0)
    {
      throw std::invalid_argument("CudaBuffer size must be greater than zero");
    }
    else
    {
      m_iSize = iSize;
      this->_allocate();
    }
    
    
  }
  
private:
  T* m_pData;
  std::size_t m_iSize;
  void _allocate();    
};


template <typename T>
void CudaBuffer<T>::_allocate() {
  
  CUDA_CHECK(cudaMalloc((void**)&m_pData, m_iSize));
}

template <typename T>
void CudaBuffer<T>::copyFromHost(const T* pData_h) {
  
  CUDA_CHECK(cudaMemcpy(m_pData, pData_h, m_iSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaGetLastError());
}



template <typename T>
void CudaBuffer<T>::copyFromDevice(T* pData_h) const {

  CUDA_CHECK(cudaMemcpy(pData_h, m_pData , m_iSize, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaGetLastError());
}


#endif 
