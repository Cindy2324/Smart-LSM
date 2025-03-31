# Install script for directory: /Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Library/Developer/CommandLineTools/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/cmake-build-debug/third_party/llama.cpp/ggml/src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/cmake-build-debug/bin/libggml.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/cmake-build-debug/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.dylib")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml-cpu.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml-alloc.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml-backend.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml-blas.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml-cann.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml-cpp.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml-cuda.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml-kompute.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml-opt.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml-metal.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml-rpc.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml-sycl.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/ggml-vulkan.h"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/third_party/llama.cpp/ggml/include/gguf.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/cmake-build-debug/bin/libggml-base.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.dylib")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ggml" TYPE FILE FILES
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/cmake-build-debug/third_party/llama.cpp/ggml/ggml-config.cmake"
    "/Users/cindy/Desktop/ads/practice/lab-lsm-tree-handout/cmake-build-debug/third_party/llama.cpp/ggml/ggml-version.cmake"
    )
endif()

