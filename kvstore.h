#pragma once

#include "kvstore_api.h"
#include "skiplist.h"
#include "sstable.h"
#include "sstablehead.h"
#include "hnsw_index.h"
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
struct HNSWNode {
    uint64_t key;
    std::unordered_map<int, std::vector<uint64_t>> neighbor;
};

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
    void insert_hnsw_node(const std::uint64_t& key, const std::vector<float>& vec);
    std::vector<uint64_t> search_layer(uint64_t ep_id, const std::vector<float> &query_vec, int level, int ef);
    std::vector<std::pair<std::uint64_t, std::string>> search_knn_hnsw(std::string query, int k);
    //uint64_t searchLayersGreedy(uint64_t epid, const std::vector<float> &vec, int fromLevel, int toLevel) const;
};
