cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
project(ImgProcLab 
        VERSION 0.0.1 
        DESCRIPTION "Image Processing Lab"  
	LANGUAGES CUDA CXX C)

message("CMake Version: ${CMAKE_VERSION}")
message("OS: ${CMAKE_SYSTEM_NAME}")



set(ARCH 32)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  Set(ARCH 64)
endif()



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES native)

set(OUTPUT_TARGET_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}/${ARCH}/${CMAKE_BUILD_TYPE}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_TARGET_DIRECTORY}/libs/static") # static
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_TARGET_DIRECTORY}/libs/dynamic") # dynamic
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_TARGET_DIRECTORY}")

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(ImgProcLab STATIC
           
			  ${CMAKE_CURRENT_SOURCE_DIR}/src/filters/GaussianFilter.cpp
			  #${CMAKE_CURRENT_SOURCE_DIR}/include/ImgProc/filters/GaussianFilter.hpp

			  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/gaussian/GaussianV1.cu
			  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/gaussian/GaussianV2.cu
              ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/gaussian/GaussianV3.cu
			  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/gaussian/GaussianPrepare.cu

			  #${CMAKE_CURRENT_SOURCE_DIR}/include/ImgProc/cuda/gaussian/GaussianCuda.hpp
			  #${CMAKE_CURRENT_SOURCE_DIR}/include/ImgProc/cuda/gaussian/GaussianData.hpp

			  #${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/common/CudaUtils.hpp
			  #${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/common/CudaBuffer.hpp
			  #${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/common/CudaExecution.hpp
			  #${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/common/CudaExecutionContext.hpp
			  #${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/common/OpenCVExecution.hpp

              #${CMAKE_CURRENT_SOURCE_DIR}/include/common/ImgProcLabUtils.hpp
			  #${CMAKE_CURRENT_SOURCE_DIR}/include/common/GaussianUtils.hpp

			  )

# to allow nvcc see a variable create in translation(.cu)  unit in some other translation unit(.cu) 
set_target_properties(ImgProcLab PROPERTIES CUDA_SEPARABLE_COMPILATION ON)			  
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

if(PROFILE_CUDA AND CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Release mode and cuda profiling enabled")
  target_compile_definitions(ImgProcLab PUBLIC PROFILE_CUDA)
  target_compile_options(ImgProcLab PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
 message(STATUS "Release mode enabled")

elseif(CMAKE_BUILD_TYPE STREQUAL "Debug") 
  message(STATUS "Debug mode enabled")
  target_compile_definitions(ImgProcLab PUBLIC $<$<CONFIG:Debug>:IMGPROCLAB_DEBUG>)
  target_compile_options(ImgProcLab PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>  
                                            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
                                            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
					                        $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>)
  
endif()

#sudo /usr/local/cuda-13.4/bin/ncu \
#    --set full \
#    --kernel-name regex:GaussianFilterKernel \
#    --launch-skip 10 \
#    --launch-count 1 \
#    --export gaussian_full \
#    ./ImgProcLab \
#    /home/kuntima/Workspace/Lab/OpenCV/images/model.png
#/usr/local/cuda-13.4/bin/ncu-ui gaussian_full.ncu-repz
