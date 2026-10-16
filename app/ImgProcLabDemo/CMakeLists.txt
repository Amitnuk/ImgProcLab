
add_executable(ImgProcLabDemo 
               ${CMAKE_CURRENT_SOURCE_DIR}/src/EntryPoint.cpp

               ${CMAKE_CURRENT_SOURCE_DIR}/src/ArgumentParser.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/include/ArgumentParser.hpp

               
 )

target_include_directories(ImgProcLabDemo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries( ImgProcLabDemo PRIVATE ImgProcLab)