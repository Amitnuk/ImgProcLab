add_executable(ImgProcLabDemo 
               ${CMAKE_CURRENT_SOURCE_DIR}/src/EntryPoint.cpp

               ${CMAKE_CURRENT_SOURCE_DIR}/src/ArgumentParser.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/include/ArgumentParser.hpp

               ${CMAKE_CURRENT_SOURCE_DIR}/src/AlgorithmRunner.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/include/AlgorithmRunner.hpp
               
               ${CMAKE_CURRENT_SOURCE_DIR}/src/BenchmarkWriter.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/include/BenchmarkWriter.hpp

               ${CMAKE_CURRENT_SOURCE_DIR}/include/BenchmarkResult.hpp
               )

target_include_directories(ImgProcLabDemo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries( ImgProcLabDemo PRIVATE ImgProcLab)


set(BENCHMARK_FOLDER_NAME "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/results")

configure_file( "Config.h.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h" )