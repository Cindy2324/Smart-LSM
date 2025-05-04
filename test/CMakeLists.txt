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

add_executable(Vector_Persistent_Test_Phase1
        Vector_Persistent_Test_Phase1.cpp
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

add_executable(Vector_Persistent_Test_Phase2
        Vector_Persistent_Test_Phase2.cpp
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

add_executable(HNSW_Persistent_Test_Phase1
        HNSW_Persistent_Test_Phase1.cpp
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

add_executable(HNSW_Persistent_Test_Phase2
        HNSW_Persistent_Test_Phase2.cpp
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

add_executable(HNSW_Delete_Test
        HNSW_Delete_Test.cpp
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

target_include_directories(Vector_Persistent_Test_Phase1 PUBLIC
        ${CMAKE_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/../third_party/llama.cpp/common
        ${CMAKE_SOURCE_DIR}/../third_party/llama.cpp/include
)

target_include_directories(Vector_Persistent_Test_Phase2 PUBLIC
        ${CMAKE_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/../third_party/llama.cpp/common
        ${CMAKE_SOURCE_DIR}/../third_party/llama.cpp/include
)

target_include_directories(HNSW_Persistent_Test_Phase1 PUBLIC
        ${CMAKE_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/../third_party/llama.cpp/common
        ${CMAKE_SOURCE_DIR}/../third_party/llama.cpp/include
)

target_include_directories(HNSW_Persistent_Test_Phase2 PUBLIC
        ${CMAKE_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/../third_party/llama.cpp/common
        ${CMAKE_SOURCE_DIR}/../third_party/llama.cpp/include
)

target_include_directories(HNSW_Delete_Test PUBLIC
        ${CMAKE_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/../third_party/llama.cpp/common
        ${CMAKE_SOURCE_DIR}/../third_party/llama.cpp/include
)