
add_executable(ImgProcLabDemo ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp )

target_include_directories(ImgProcLabDemo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries( ImgProcLabDemo PRIVATE ImgProcLab)