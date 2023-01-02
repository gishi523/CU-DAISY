cmake_minimum_required(VERSION 3.18)

# create project
project(CU-DAISY LANGUAGES CXX CUDA)

option(WITH_OPENCV_DAISY "compare with OpenCV DAISY" OFF)
set(CUDA_ARCHS "52;61;72;75;86" CACHE STRING "List of architectures to generate device code for")

message(STATUS "WITH_OPENCV_DAISY : ${WITH_OPENCV_DAISY}")
message(STATUS "CUDA_ARCHS        : ${CUDA_ARCHS}")

file(GLOB srcs *.cpp *.h* *.cu)

# dependent packages
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(WITH_OPENCV_DAISY)
	find_package(OpenCV REQUIRED xfeatures2d)
endif()

# target configuration
add_executable(cudaisy ${srcs})
target_include_directories(cudaisy PRIVATE ${OpenCV_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
target_compile_features(cudaisy PRIVATE cxx_std_17)
target_link_libraries(cudaisy ${OpenCV_LIBS} CUDA::cudart)
set_target_properties(cudaisy PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHS})

target_compile_options(cudaisy PRIVATE
$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wall -O3 -march=native>
$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wall -O3>
$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4819>
$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-lineinfo -Xcompiler="\"/wd4819 /wd4244\"">
)

target_compile_definitions(cudaisy PRIVATE
$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:_SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING>
$<$<BOOL:${WITH_OPENCV_DAISY}>:WITH_OPENCV_DAISY>
)
