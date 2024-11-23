cmake_minimum_required(VERSION 3.28.0)
project(cugemm LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

if(CMAKE_CUDA_COMPILER)
    message("CUDA found.")
    # message(${CMAKE_CUDA_COMPILER})
else()
    message(FATAL_ERROR "CUDA not found.")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")


set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cu main.cpp)

target_link_libraries(cu ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(cu -lcublas)

# OpenBLAS detection & linking
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)

if(BLAS_FOUND)
    message("OpenBLAS found.")
    include_directories(/usr/local/include/openblas/)
    target_link_libraries(cu ${BLAS_LIBRARIES})
else()
    message(FATAL_ERROR "OpenBLAS not found.")
endif()


