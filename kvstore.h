#pragma once

#include "kvstore_api.h"
#include "skiplist.h"
#include "sstable.h"
#include "sstablehead.h"
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cmath>


#include <map>
#include <set>

struct HNSWNode {
    uint64_t id;
    std::vector<uint64_t> neighbors;
};

class KVStore : public KVStoreAPI {
    // You can add your implementation here
private:
    skiplist *s = new skiplist(0.5); // memtable
    // std::vector<sstablehead> sstableIndex;  // sstable的表头缓存

    std::vector<sstablehead> sstableIndex[15]; // the sshead for each level

    int totalLevel = -1; // 层数

    std::unordered_map<uint64_t, std::vector<float>> vectorStore;

    const int m_L = 6;
    const int M = 6;
    const int M_max = 8;
    int max_level = -1;
    uint64_t entry_point = UINT64_MAX;
    const static int efConstruction = 30;

    std::vector<std::unordered_map<uint64_t, HNSWNode>> hnsw_levels;

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

    float cosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2);

    void loadVectorsFromSSTables();

    void insert_hnsw_node(const std::uint64_t& key, const std::vector<float>& vec);

    std::vector<uint64_t> search_layer(uint64_t ep_id, const std::vector<float> &query_vec, int level, int ef);

    int random_level() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(0.0, 1.0);
        double randValue = std::max(dis(gen), 1e-10); // 防止 log(0)
        int level = static_cast<int>(-std::log(randValue) * m_L);
        return std::min(level, m_L); // 设置最大层数为 m_L
    }

    std::vector<std::pair<std::uint64_t, std::string>> search_knn_hnsw(std::string query, int k);
};
