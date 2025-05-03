set(CMAKE_BUILD_TYPE Debug)

add_executable(E2E_test
        E2E_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../kvstore.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../skiplist.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../sstable.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../bloom.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../MurmurHash3.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../sstablehead.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../hnsw_index.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../hnsw_index.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../embedding/embedding.cc
)

target_link_libraries(E2E_test PUBLIC embedding llama)


target_include_directories(E2E_test PUBLIC
        ${CMAKE_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/../third_party/llama.cpp/common
        ${CMAKE_SOURCE_DIR}/../third_party/llama.cpp/include
)

