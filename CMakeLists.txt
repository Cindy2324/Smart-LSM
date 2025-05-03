cmake_minimum_required(VERSION 3.10)
project(LSM_TREE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

add_executable(correctness correctness.cc kvstore_api.h kvstore.h
        kvstore.cc skiplist.cpp skiplist.h sstable.cpp sstable.h
        bloom.cpp bloom.h MurmurHash3.h utils.h test.h
        sstablehead.cpp sstablehead.h hnsw_index.cpp hnsw_index.h)

add_executable(persistence persistence.cc kvstore_api.h kvstore.h kvstore.cc
        skiplist.cpp skiplist.h sstable.cpp sstable.h
        bloom.cpp bloom.h MurmurHash3.h utils.h test.h
        sstablehead.cpp sstablehead.h hnsw_index.cpp hnsw_index.h)

add_executable(efficiency efficiency.cc kvstore_api.h kvstore.h kvstore.cc
        skiplist.cpp skiplist.h sstable.cpp sstable.h
        bloom.cpp bloom.h MurmurHash3.h utils.h test.h
        sstablehead.cpp sstablehead.h hnsw_index.cpp hnsw_index.h)


add_subdirectory(third_party/llama.cpp)

add_subdirectory(embedding)

add_subdirectory(test)


target_link_libraries(correctness PUBLIC llama embedding)
target_link_libraries(persistence PUBLIC llama embedding)
target_link_libraries(efficiency PUBLIC llama embedding)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)