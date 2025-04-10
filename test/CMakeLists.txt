add_executable(Embedding_Test Embedding_Test.cpp)

target_link_libraries(Embedding_Test PUBLIC embedding)

include_directories(${CMAKE_SOURCE_DIR})
add_executable(E2E_test E2E_test.cpp
        ../kvstore_api.h
        ../kvstore.h
        ../kvstore.cc
        ../skiplist.cpp
        ../skiplist.h
        ../sstable.cpp
        ../sstable.h
        ../bloom.cpp
        ../bloom.h
        ../MurmurHash3.h
        ../utils.h
        ../test.h
        ../sstablehead.cpp
        ../sstablehead.h)

target_link_libraries(E2E_test PUBLIC embedding)