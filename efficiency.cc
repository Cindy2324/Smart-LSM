//
// Created by Cindy on 25-3-24.
//
#include <iostream>
#include <vector>
#include <chrono>
#include <atomic>
#include "kvstore.h"  // 你的 KVStore 头文件

#define TEST_SIZE 100   // 测试的 key 数量

KVStore kv("./test");  // 初始化你的键值存储

void test_put(int start, int end, int& count, double& put_time) {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = start; i < end; i++) {
        kv.put(i, "value" + std::to_string(i));
        count++;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    put_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "PUT 完成, 用时: "
              << std::chrono::duration<double>(end_time - start_time).count()
              << " 秒" << std::endl;
    //std::cout<<count<<std::endl;
}

void test_get(int start, int end, int& count, double& get_time) {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = start; i < end; i++) {
        kv.get(i);
        count++;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    get_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "GET 完成, 用时: "
              << std::chrono::duration<double>(end_time - start_time).count()
              << " 秒" << std::endl;
}

void test_del(int start, int end, int& count, double& del_time) {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = start; i < end; i++) {
        kv.del(i);
        count++;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    del_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "DEL 完成, 用时: "
              << std::chrono::duration<double>(end_time - start_time).count()
              << " 秒" << std::endl;
}

void run_test() {
    int put_count = 0, get_count = 0, del_count = 0;
    double put_time, get_time, del_time = 0;
    int chunk = TEST_SIZE;

    // PUT 测试
    auto start_time = std::chrono::high_resolution_clock::now();
    test_put(0, chunk, put_count, put_time);
    auto end_time = std::chrono::high_resolution_clock::now();
    //double put_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "PUT 吞吐量: " << (put_count / put_time) << " ops/sec" << std::endl;

    // GET 测试
    start_time = std::chrono::high_resolution_clock::now();
    test_get(0, chunk, get_count, get_time);
    end_time = std::chrono::high_resolution_clock::now();
    //double get_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "GET 吞吐量: " << (get_count / get_time) << " ops/sec" << std::endl;

    // DEL 测试
    start_time = std::chrono::high_resolution_clock::now();
    test_del(0, chunk, del_count, del_time);
    end_time = std::chrono::high_resolution_clock::now();
    //double del_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "DEL 吞吐量: " << (del_count / del_time) << " ops/sec" << std::endl;
}

int main() {
    run_test();
    return 0;
}