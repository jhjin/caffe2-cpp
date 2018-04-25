cmake_minimum_required(VERSION 2.8.3)
project(caffe2)
include(ExternalProject)

find_package(PythonInterp 2.7)
find_package(PythonLibs 2.7)
find_package(CUDA)

if(NOT CUDA_FOUND)
  set(USE_CUDA OFF)
else()
  set(USE_CUDA ON)
endif()

execute_process(COMMAND "python -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())')" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)

ExternalProject_Add(${PROJECT_NAME}-ext
  PREFIX ext
  GIT_REPOSITORY http://github.com/pytorch/pytorch.git
  GIT_TAG 664fe34e0a28318947edaa562247be7b583c1ee7
  INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}

  PATCH_COMMAND git checkout .
        COMMAND git clean -f
        COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/caffe2.patch

  CMAKE_ARGS    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                -DPYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE_DIR}
                -DCUDNN_ROOT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/cuda
                -DUSE_CUDA=${USE_CUDA}
                -DUSE_OPENMP=ON
                -DUSE_LEVELDB=OFF
                -DUSE_LMDB=OFF
                -DUSE_NATIVE_ARCH=ON
                -DUSE_METAL=OFF
                -DBUILD_CUSTOM_PROTOBUF=ON
)
