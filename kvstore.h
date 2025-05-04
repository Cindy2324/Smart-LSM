#pragma once

#include "kvstore_api.h"
#include "skiplist.h"
#include "sstable.h"
#include "sstablehead.h"
#include "hnsw_index.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cmath>


#include <map>
#include <set>

// struct HNSWNode {
//     uint64_t id;
//     std::vector<uint64_t> neighbors;
// };
// struct HNSWNode {
//     uint64_t key;
//     std::unordered_map<int, std::vector<uint64_t>> neighbor;
// };

class KVStore : public KVStoreAPI {
    // You can add your implementation here
private:
    skiplist *s = new skiplist(0.5); // memtable
    // std::vector<sstablehead> sstableIndex;  // sstable的表头缓存
    std::vector<sstablehead> sstableIndex[15]; // the sshead for each level
    int totalLevel = -1; // 层数
    std::unordered_map<uint64_t, std::vector<float>> vectorStore;
    HNSWIndex hnsw_index;
    friend class HNSWIndex;
    uint64_t dim = hnsw_index.dim;
    std::vector<float> tombstone;
public:
    KVStore(const std::string &dir);
    ~KVStore();
    void put(uint64_t key, const std::string &s) override;
    std::string get(uint64_t key) override;
    bool del(uint64_t key) override;
    void reset() override;
    void scan(uint64_t key1, uint64_t key2, std::list<std::pair<uint64_t, std::string>> &list) override;
    void compaction();
    void delsstable(std::string filename);  // 从缓存中删除filename.sst， 并物理删除
    void addsstable(sstable ss, int level); // 将ss加入缓存
    std::string fetchString(std::string file, int startOffset, uint32_t len);
    std::vector<std::pair<std::uint64_t, std::string>> search_knn(std::string query, int k);
    float cosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2) const;
    void loadVectorsFromSSTables();
    void load_embedding_from_disk(const std::string &data_root);
    void save_hnsw_index_to_disk(const std::string &hnsw_data_root);
    void insert_hnsw_node(const std::uint64_t& key, const std::vector<float>& vec);
    std::vector<uint64_t> search_layer(uint64_t ep_id, const std::vector<float> &query_vec, int level, int ef = HNSWIndex::efConstruction);
    std::vector<std::pair<std::uint64_t, std::string>> search_knn_hnsw(std::string query, int k);
    std::vector<float> get_embedding_for_value(const std::string& val) {
        // 打开文件读取 cleaned_text_100k.txt 查找 val
        std::ifstream file("cleaned_text_100k.txt");
        std::string line;
        size_t line_number = 0;  // 行号从零开始

        // 查找 val 在 cleaned_text_100k.txt 中对应的行号
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (line == val) {
                // 找到对应的行号后，读取 embedding_100k.txt 中的向量
                break;
            }
            line_number++;  // 每读取一行，行号增加
        }

        // 如果没有找到，返回空向量
        if (line_number == static_cast<size_t>(-1)) {
            std::cerr << "Value not found in cleaned_text_100k.txt!" << std::endl;
            return {};
        }

        // 打开 embedding_100k.txt 文件并读取对应的向量
        std::ifstream embedding_file("embedding_100k.txt");
        for (size_t i = 0; i < line_number; ++i) {
            std::getline(embedding_file, line);  // 跳过前面的行
            std::getline(embedding_file, line);
        }

        std::vector<float> embedding;
        if (std::getline(embedding_file, line)) {
            // 去掉前后的 "[]"
            line = line.substr(1, line.size() - 2);

            // 使用 stringstream 将字符串按逗号分割
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ',')) {
                // 转换每个数值为 float 并添加到向量中
                embedding.push_back(std::stof(value));
            }
        }

        return embedding;
    }
    //uint64_t searchLayersGreedy(uint64_t epid, const std::vector<float> &vec, int fromLevel, int toLevel) const;
};
