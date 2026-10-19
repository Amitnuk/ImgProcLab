cmake_minimum_required(VERSION 3.18)

project(ImgProcLab 
        VERSION 0.0.1 
		DESCRIPTION "Image Processing Lab"  
		LANGUAGES CUDA CXX C)

message("CMake Version: ${CMAKE_VERSION}")
message("OS: ${CMAKE_SYSTEM_NAME}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(ImgProcLab STATIC
           
			  ${CMAKE_CURRENT_SOURCE_DIR}/src/filters/GaussianFilter.cpp
			  ${CMAKE_CURRENT_SOURCE_DIR}/include/ImgProc/filters/GaussianFilter.hpp

			  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/gaussian/GaussianV1.cu
			  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/gaussian/GaussianPrepare.cu

			  ${CMAKE_CURRENT_SOURCE_DIR}/include/ImgProc/cuda/gaussian/GaussianCuda.hpp
			  ${CMAKE_CURRENT_SOURCE_DIR}/include/ImgProc/cuda/gaussian/GaussianData.hpp

			  ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/common/CudaUtils.hpp
			  ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/common/CudaBuffer.hpp
			  ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/common/CudaExecutionContext.hpp
			  )

			  
target_include_directories(ImgProcLab PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ImgProcLab PUBLIC ${OpenCV_LIBS} CUDA::cudart)

#[[
target_compile_options(ImgProcLab PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>  
                                          $<$<COMPILE_LANGUAGE:CXX>:-Wall> 
										  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
										  $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>)
#]]

set(TARGET_APPLICATION app/ImgProcLabDemo)

add_subdirectory(${TARGET_APPLICATION})

#option(PROFILE_CUDA "Enable CUDA profilling mode" OFF)

#if(PROFILE_CUDA)
#	target_compile_definitions(ImgProcLab PRIVATE PROFILE_CUDA)
#endif()

#sudo /usr/local/cuda-13.4/bin/ncu \
#    --set full \
#    --kernel-name regex:GaussianFilterKernel \
#    --launch-skip 10 \
#    --launch-count 1 \
#    --export gaussian_full \
#    ./ImgProcLab \
#    /home/kuntima/Workspace/Lab/OpenCV/images/model.png
#/usr/local/cuda-13.4/bin/ncu-ui gaussian_full.ncu-repz