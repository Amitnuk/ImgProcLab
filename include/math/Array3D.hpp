#ifndef ARRAY_HPP
#define ARRAY_HPP
#include <vector>
#include <cstddef>
#include <stdexcept>

template <typename T>
class Array {

public:
  Array():m_iHeight(0), m_iWidth(0), m_iDepth(0) {}

  explicit Array(std::size_t iHeight, std::size_t iWidth, std::size_t iDepth = 1);
  T& operator()(std::size_t iRow, std::size_t iCol, std::size_t iDepth = 0);
  const T& operator()(std::size_t iRow, std::size_t iCol, std::size_t iDepth = 0) const;

  Array(const Array&) = delete;
  Array& operator=(const Array&)= delete;

  Array(Array&&) noexcept = default;
  Array& operator=(Array&&) noexcept = default;

  inline T* data() {return m_vData.data();}
  inline const T* data() const {return m_vData.data();}
  inline std::size_t height() const {return  m_iHeight;}
  inline std::size_t width()  const { return m_iWidth; }
  inline std::size_t depth()  const { return m_iDepth; }
  inline std::size_t size()   const {return m_iDepth*m_iHeight*m_iWidth ;}
  
  
private:
  std::size_t m_iHeight;
  std::size_t m_iWidth;
  std::size_t m_iDepth;  
  std::vector<T> m_vData; // without the destructor, the compiler generates a move semantic constructor
}  ;


template <typename T>
Array<T>::Array(std::size_t iHeight, std::size_t iWidth, std::size_t iDepth /*= 1*/)

    : m_iHeight(iHeight)
    , m_iWidth(iWidth)
    , m_iDepth(iDepth)
    , m_vData(m_iDepth*m_iHeight*m_iWidth)      {

  
}

template <typename T>
T &Array<T>::operator()(std::size_t iRow, std::size_t iCol, std::size_t iDepth) {

  if(iRow >= m_iHeight || iCol >= m_iWidth || iDepth >= m_iDepth)
  {
    throw std::out_of_range("Array index out of bounds");
  }

  return m_vData[iDepth*m_iWidth*m_iHeight + iRow*m_iWidth + iCol];
}

template <typename T>
const T &Array<T>::operator()(std::size_t iRow, std::size_t iCol, std::size_t iDepth) const {

  if(iRow >= m_iHeight || iCol >= m_iWidth || iDepth >= m_iDepth)
  {
    throw std::out_of_range("Array index out of bounds");
  }

  return m_vData[iDepth*m_iWidth*m_iHeight + iRow*m_iWidth + iCol];
}
#endif
