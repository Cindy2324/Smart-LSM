
#pragma once
#include "kvstore.h"
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <random>


class HNSWIndex {
public:
    HNSWIndex();

    ~HNSWIndex();

    void insertNode(KVStore& store, uint64_t key, const std::vector<float>& vec);
    std::vector<uint64_t> search_layer(KVStore& store, uint64_t ep_id, const std::vector<float> &query_vec, int level, int ef = efConstruction);
    void::reset() {
        nodes.clear();
        max_level = -1;
        entry_point = UINT64_MAX;
    }
    void del(uint64_t key);

    const int m_L = 5;
    const int M = 10;
    const int M_max = 15;
    const static int efConstruction = 30;

    int max_level = -1;
    uint64_t entry_point = UINT64_MAX;
private:
    struct HNSWNode {
        uint64_t key;
        std::unordered_map<int, std::vector<uint64_t>> neighbor;
    };

    int random_level() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(0.0, 1.0);
        double randValue = std::max(dis(gen), 1e-10); // 防止 log(0)
        int level = static_cast<int>(-std::log(randValue) * m_L);
        //double randValue = dis(gen);
        //int level = static_cast<int>(randValue * m_L);
        return std::min(level, m_L); // 设置最大层数为 m_L
    }

    float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) const;
    std::unordered_map<uint64_t, HNSWNode> nodes;
};
