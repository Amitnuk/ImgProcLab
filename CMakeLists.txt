cmake_minimum_required(VERSION 3.18)

project(ImgProcLab LANGUAGES CUDA CXX C)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(OpenCV REQUIRED)
add_executable(ImgProcLab
              ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp 

			  ${CMAKE_CURRENT_SOURCE_DIR}/include/ImgProc/core/BaseFilter.hpp

              ${CMAKE_CURRENT_SOURCE_DIR}/src/filters/BoxFilter.cu
			  ${CMAKE_CURRENT_SOURCE_DIR}/include/ImgProc/filters/BoxFilter.hpp
			  ${CMAKE_CURRENT_SOURCE_DIR}/src/filters/GaussianFilter.cu
			  ${CMAKE_CURRENT_SOURCE_DIR}/include/ImgProc/filters/GaussianFilter.hpp
			  ${CMAKE_CURRENT_SOURCE_DIR}/src/filters/MedianFilter.cu
			  ${CMAKE_CURRENT_SOURCE_DIR}/include/ImgProc/filters/MedianFilter.hpp


			  ${CMAKE_CURRENT_SOURCE_DIR}/src/EdgeDetection/SobelFilter.cu
			  ${CMAKE_CURRENT_SOURCE_DIR}/include/ImgProc/EdgeDetection/SobelFilter.hpp


			  ${CMAKE_CURRENT_SOURCE_DIR}/src/Thresholding/BinaryThreshold.cu
			  ${CMAKE_CURRENT_SOURCE_DIR}/include/ImgProc/Thresholding/BinaryThreshold.hpp

			  ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/CudaUtils.hpp
			  ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/CudaBuffer.hpp)

			  
target_include_directories(ImgProcLab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ImgProcLab PRIVATE ${OpenCV_LIBS})

target_compile_options(ImgProcLab PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>)

