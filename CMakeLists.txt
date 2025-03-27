
cmake_minimum_required(VERSION 3.30)
project(lab-lsm-tree-handout)
set(CMAKE_CXX_STANDARD 20)
add_executable(correctness correctness.cc kvstore_api.h kvstore.h
        kvstore.cc skiplist.cpp skiplist.h sstable.cpp sstable.h
        bloom.cpp bloom.h MurmurHash3.h utils.h test.h
        sstablehead.cpp sstablehead.h
        kvstore.cc)

add_executable(persistence persistence.cc kvstore_api.h kvstore.h kvstore.cc
        skiplist.cpp skiplist.h sstable.cpp sstable.h
        bloom.cpp bloom.h MurmurHash3.h utils.h test.h
        sstablehead.cpp sstablehead.h
        kvstore.cc)

add_executable(efficiency efficiency.cc kvstore_api.h kvstore.h kvstore.cc
        skiplist.cpp skiplist.h sstable.cpp sstable.h
        bloom.cpp bloom.h MurmurHash3.h utils.h test.h
        sstablehead.cpp sstablehead.h
        kvstore.cc)