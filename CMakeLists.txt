cmake_minimum_required(VERSION 2.8)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall")
endif()

# OpenCV
find_package(OpenCV REQUIRED)
if(WITH_OPENCV_DAISY)
	message("-- WITH_OPENCV_DAISY ON")
	find_package(OpenCV REQUIRED xfeatures2d)
	ADD_DEFINITIONS(-DWITH_OPENCV_DAISY)
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -lineinfo -gencode arch=compute_61,code=sm_61")
file(GLOB srcs ./*.cpp ./*.h* ./*.cu)
cuda_add_executable(cudaisy ${srcs})
target_link_libraries(cudaisy ${OpenCV_LIBS})