#pragma once
#include "utils.h"
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <random>
#include <map>

class KVStore;
static uint64_t ID = 0;

class HNSWIndex {
public:
    HNSWIndex();

    ~HNSWIndex();

    void insertNode(KVStore& store, uint64_t key, const std::vector<float>& vec);
    std::vector<uint64_t> search_layer(KVStore& store, uint64_t ep_id, const std::vector<float> &query_vec, int level, int ef);
    void reset();
    void del(uint64_t key);
    void append_embeddings_to_disk(const std::map<uint64_t, std::vector<float>> &batch);

    int m_L = 5;
    int M = 10;
    int M_max = 15;
    static int efConstruction;

    int max_level = -1;
    uint64_t entry_point = UINT64_MAX;

    const uint64_t dim = 768;
private:
    struct HNSWNode {
        uint64_t key;
        std::unordered_map<int, std::vector<uint64_t>> neighbor;
    };
    std::unordered_set<uint64_t> deleted_nodes;
    float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) const;
    std::unordered_map<uint64_t, HNSWNode> nodes;
    //std::unordered_map<int, HNSWNode*> node_id_map;

    std::string embedding_dir = "./embedding_data";
    std::string embedding_file = "./embedding_data/embedding_vectors.bin";

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

    bool initialized = false;
    void init_embedding_file() {
        if (!initialized) {
            if (!utils::dirExists(embedding_dir)) {
                utils::mkdir(embedding_dir.c_str());
            }

            if (!std::ifstream(embedding_file)) {
                std::ofstream ofs(embedding_file, std::ios::binary);
                uint64_t d = dim;
                ofs.write(reinterpret_cast<const char*>(&d), sizeof(uint64_t)); // 写入维度信息
                ofs.close();
            }
            initialized = true;
        }
    }
};

int HNSWIndex::efConstruction = 30;
