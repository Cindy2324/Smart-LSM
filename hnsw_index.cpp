#include "hnsw_index.h"
#include "kvstore.h"
#include <cstdlib>
#include <cmath>
#include <limits>
#include <queue>
#include <algorithm>

HNSWIndex::HNSWIndex() {
    init_embedding_file();
}
HNSWIndex::~HNSWIndex() {
    nodes.clear();
}

void HNSWIndex::reset() {
    nodes.clear();
    deleted_nodes.clear();
    hnsw_header.entry_point = UINT64_MAX;
    hnsw_header.max_level = -1;
    initialized = false;
}

void HNSWIndex::del(uint64_t key) {
    // 没找到则忽略
    if (key_to_ids.find(key) == key_to_ids.end()) {
        return;
    }
    uint64_t id = key_to_ids.at(key);
    deleted_nodes.insert(id);
    key_to_ids.erase(key);
}

void HNSWIndex::append_embeddings_to_disk(const std::map<uint64_t, std::vector<float>> &batch)  {
    std::ofstream ofs(embedding_file, std::ios::binary | std::ios::app);
    for (const auto &[key, vec] : batch) {
        if (vec.size() != hnsw_header.dim) {
            throw std::runtime_error("Embedding dimension mismatch");
        }
        ofs.write(reinterpret_cast<const char*>(&key), sizeof(uint64_t));
        ofs.write(reinterpret_cast<const char*>(vec.data()), sizeof(float) * hnsw_header.dim);
    }
    ofs.close();
}

void HNSWIndex::save_hnsw_index_to_disk(const std::string &hnsw_data_root) {
    std::string global_header_path = hnsw_data_root + "/global_header.bin";
    std::ofstream global_header_out(global_header_path, std::ios::binary);
    global_header_out.write(reinterpret_cast<char*>(&hnsw_header), sizeof(hnsw_header));
    // 写入 key_to_ids 大小
    uint64_t map_size = key_to_ids.size();
    global_header_out.write(reinterpret_cast<char*>(&map_size), sizeof(map_size));

    // 依次写入每个 key-value 对
    for (const auto& [key, id] : key_to_ids) {
        global_header_out.write(reinterpret_cast<const char*>(&key), sizeof(key));
        global_header_out.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }
    global_header_out.close();

    std::string deleted_path = hnsw_data_root + "/deleted_nodes.bin";
    std::ofstream deleted_out(deleted_path, std::ios::binary);
    uint64_t deleted_size = deleted_nodes.size();
    deleted_out.write(reinterpret_cast<char*>(&deleted_size), sizeof(uint64_t));
    for (uint64_t node_id : deleted_nodes) {
        deleted_out.write(reinterpret_cast<char*>(&node_id), sizeof(uint64_t));
    }
    deleted_out.close();

    if (!utils::dirExists(hnsw_data_root+ "/nodes")) {
        utils::mkdir((hnsw_data_root+ "/nodes").c_str());
    }

    for (const auto& [node_id, node] : nodes) {
        std::string node_dir = hnsw_data_root + "/nodes/" + std::to_string(node_id);
        if (!utils::dirExists(node_dir)) {
            utils::mkdir(node_dir.c_str());
        }
        if (!utils::dirExists(node_dir + "/edges")) {
            utils::mkdir((node_dir + "/edges").c_str());
        }

        // 保存 header.bin
        std::string header_path = node_dir + "/header.bin";
        std::ofstream header_out(header_path, std::ios::binary);
        //header_out.write(reinterpret_cast<char*>(const_cast<uint64_t*>(&node_id)), sizeof(uint64_t));
        header_out.write(reinterpret_cast<char*>(const_cast<uint64_t*>(&node.key)), sizeof(uint64_t));
        header_out.close();

        // 保存每一层邻接边 edges/<level>.bin
        for (const auto& [level, neighbors] : nodes[node_id].neighbor) {
            std::string edge_path = node_dir + "/edges/" + std::to_string(level) + ".bin";
            std::ofstream edge_out(edge_path, std::ios::binary);
            uint32_t neighbor_count = neighbors.size();
            //fprintf(stderr, "node_id: %llu, level: %d, neighbor_count: %u\n", static_cast<unsigned long long>(node_id), level, neighbor_count);
            edge_out.write(reinterpret_cast<char*>(&neighbor_count), sizeof(uint32_t));
            for (uint64_t neighbor_id : neighbors) {
                edge_out.write(reinterpret_cast<char*>(&neighbor_id), sizeof(uint64_t));
            }
            edge_out.close();
        }
    }
    int debug_dummy = 0;
}

void HNSWIndex::load_hnsw_index_to_disk(const std::string &hnsw_data_root) {
    initialized = true;
    nodes.clear();
    deleted_nodes.clear();
    key_to_ids.clear();

    // 1. 加载 global_header.bin
    std::string header_path = hnsw_data_root + "/global_header.bin";
    std::ifstream header_in(header_path, std::ios::binary);
    if (!header_in) throw std::runtime_error("Cannot open global_header.bin");
    header_in.read(reinterpret_cast<char*>(&hnsw_header), sizeof(HNSWGlobalHeader));
    uint64_t map_size;
    header_in.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));
    for (uint64_t i = 0; i < map_size; ++i) {
        uint64_t key, id;
        header_in.read(reinterpret_cast<char*>(&key), sizeof(key));
        header_in.read(reinterpret_cast<char*>(&id), sizeof(id));
        key_to_ids[key] = id;
    }
    //fprintf(stderr, "hnsw_header: %u %u %u %u %u %u %llu\n", hnsw_header.M, hnsw_header.M_max, hnsw_header.efConstruction, hnsw_header.m_L, hnsw_header.max_level, hnsw_header.dim, hnsw_header.entry_point);
    header_in.close();

    // 2. 加载 deleted_nodes.bin
    std::string deleted_path = hnsw_data_root + "/deleted_nodes.bin";
    std::ifstream deleted_in(deleted_path, std::ios::binary);
    if (deleted_in) {
        uint64_t count;
        deleted_in.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));
        for (uint64_t i = 0; i < count; ++i) {
            uint64_t id;
            deleted_in.read(reinterpret_cast<char*>(&id), sizeof(uint64_t));
            deleted_nodes.insert(id);
        }
        deleted_in.close();
    }

    // 3. 扫描 nodes 目录
    std::string nodes_dir = hnsw_data_root + "/nodes";
    std::vector<std::string> entries;
    utils::scanDir(nodes_dir, entries); // 只列出不以 '.' 开头的项

    // 3.1 排序为从小到大的 ID 顺序
    std::vector<uint64_t> node_ids;
    for (const auto &entry : entries) {
        try {
            node_ids.push_back(std::stoull(entry));
        } catch (...) {
            continue; // 忽略无法解析为整数的文件夹名
        }
    }
    std::sort(node_ids.begin(), node_ids.end(), std::greater<uint64_t>());
    if (!node_ids.empty()) {
        ID = node_ids.back() + 1;
    } else {
        ID = 0;
    }

    // 3.2 逐个读取每个节点的数据
    for (uint64_t node_id : node_ids) {
        nodes[node_id] = HNSWNode{UINT64_MAX, {}};
        std::string node_path = nodes_dir + "/" + std::to_string(node_id);

        // 4.1 读取 header.bin 中的 key
        std::ifstream hfs(node_path + "/header.bin", std::ios::binary);
        if (!hfs.is_open()) {
            throw std::runtime_error("Failed to open header.bin for node " + std::to_string(node_id));
        }
        hfs.read(reinterpret_cast<char *>(&nodes[node_id].key), sizeof(uint64_t));
        //fprintf(stderr, "node_id: %llu, key: %llu\n", static_cast<unsigned long long>(node_id), static_cast<unsigned long long>(nodes[node_id].key));
        hfs.close();

        // 4.2 读取 edges 下的所有邻接表文件
        std::vector<std::string> edge_files;
        std::string edge_dir = node_path + "/edges";
        utils::scanDir(edge_dir, edge_files);

        for (const auto &file : edge_files) {
            //fprintf(stderr, "file: %s\n", file.c_str());
            if (file.size() <= 4 || file.substr(file.size() - 4) != ".bin") continue;
            std::string level_str = file.substr(0, file.size() - 4);
            int level;
            try {
                level = std::stoi(level_str);
                //fprintf(stderr, "level: %d\n", level);
            } catch (...) {
                continue;
            }

            std::ifstream in(edge_dir + "/" + file, std::ios::binary);
            if (!in) {
                fprintf(stderr, "Failed to open edge file: %s\n", (edge_dir + "/" + file).c_str());
                continue;
            }
            uint32_t count;
            in.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
            for (uint64_t i = 0; i < count; ++i) {
                uint64_t id;
                if (!in.read(reinterpret_cast<char*>(&id), sizeof(uint64_t))) {
                    // 处理文件读取失败（可选打印日志或退出）
                    fprintf(stderr, "Failed to read neighbor ID from file: %s\n", (edge_dir + "/" + file).c_str());
                    break;
                }
                nodes[node_id].neighbor[level].push_back(id);
            }
        }
    }
    int debug_dummy = 0;
}

float HNSWIndex::cosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2) const {
    double dotProduct = 0.0;
    double magnitudeV1 = 0.0;
    double magnitudeV2 = 0.0;

    for (size_t i = 0; i < v1.size(); ++i) {
        dotProduct += v1[i] * v2[i];
        magnitudeV1 += v1[i] * v1[i];
        magnitudeV2 += v2[i] * v2[i];
    }

    double magnitude = std::sqrt(magnitudeV1) * std::sqrt(magnitudeV2);

    if (dotProduct == 0.0 || magnitude == 0.0) {
        if (dotProduct == 0.0 && magnitude == 0.0) {
            return 1.0f;
        }
        return 0.0f;
    }

    return dotProduct / magnitude;
}


void HNSWIndex::insertNode(KVStore& store, uint64_t key, const std::vector<float>& vec) {
    int level = random_level();
    uint64_t id = ID;
    ID++;
    if (key_to_ids.count(key) && key_to_ids[key] != id) {
        deleted_nodes.insert(key_to_ids[key]);
    }
    key_to_ids[key]=id;
    nodes[id] = HNSWNode{key, {}};
    if (hnsw_header.entry_point == UINT64_MAX) { // First node
        hnsw_header.entry_point = id;
        hnsw_header.max_level = level;
        return;
    }

     // if not the first node
    uint64_t ep = hnsw_header.entry_point;
    for (int l = hnsw_header.max_level; l > level; --l) {
        auto ep_neighbors = search_layer(store, ep, vec, l, 1);
          if (!ep_neighbors.empty()) ep = ep_neighbors[0];
    }
    //uint64_t ep = searchLayersGreedy(entry_point, vec, max_level, level + 1);

    for (int l = std::min(level, static_cast<int>(hnsw_header.max_level)); l >= 0; --l) {
        auto neighbors = search_layer(store, ep, vec, l, hnsw_header.efConstruction);
        std::vector<std::pair<float, uint64_t>> scored;
        for (uint64_t nid : neighbors)
            scored.emplace_back(cosineSimilarity(vec, store.vectorStore[nodes[nid].key]), nid);
        std::partial_sort(scored.begin(), scored.begin() + std::min(static_cast<int>(M), (int)scored.size()), scored.end(), std::greater<>());

        std::vector<uint64_t> selected;
        selected.reserve(std::min(static_cast<int>(M), (int)scored.size()));
        for (int i = 0; i < std::min(static_cast<int>(M), (int)scored.size()); ++i)
            selected.push_back(scored[i].second);

        for (uint64_t neighbor : selected) {
            nodes[neighbor].neighbor[l].push_back(id);
            nodes[id].neighbor[l].push_back(neighbor);

            if (nodes[neighbor].neighbor[l].size()> hnsw_header.M_max) {
                auto &neigh_vecs = nodes[neighbor].neighbor[l];
                std::vector<std::pair<float, uint64_t>> dist_list;
                for (uint64_t nid : neigh_vecs)
                    dist_list.emplace_back(cosineSimilarity(store.vectorStore[nodes[nid].key], store.vectorStore[nodes[neighbor].key]), nid);
                std::partial_sort(dist_list.begin(), dist_list.begin() + hnsw_header.M_max, dist_list.end(), std::greater<>());
                uint64_t far_node = dist_list.back().second;

                // 从neighbor的邻居列表中移除far_node
                neigh_vecs.erase(std::remove(neigh_vecs.begin(), neigh_vecs.end(), far_node), neigh_vecs.end());
                // 从far_node的邻居列表中移除neighbor
                auto &far_node_neighbors = nodes[far_node].neighbor[l];
                far_node_neighbors.erase(std::remove(far_node_neighbors.begin(), far_node_neighbors.end(), neighbor), far_node_neighbors.end());
            }
        }
        ep = selected.empty() ? ep : selected[0];
    }
}

std::vector<uint64_t> HNSWIndex::search_layer(KVStore& store, uint64_t ep_id, const std::vector<float> &query_vec, int level, int ef) {
    // ep_id为起始点，query_vec为要搜索的点的vector，level为当前搜索的层，efConstruction为维护当前层搜索到的与q最近邻的个数
    std::priority_queue<std::pair<float, uint64_t>, std::vector<std::pair<float, uint64_t>>, std::greater<>> top_candidates; // 最小堆
    std::queue<uint64_t> bfs_queue;
    std::unordered_set<uint64_t> visited;
    std::unordered_map<uint64_t, float> sim_cache; // 缓存相似度计算结果

    // 检查起始节点是否存在
    if (store.vectorStore.find(nodes[ep_id].key) == store.vectorStore.end()) return {};

    // 初始化
    float ep_sim = cosineSimilarity(store.vectorStore[nodes[ep_id].key], query_vec);
    top_candidates.emplace(ep_sim, ep_id);
    bfs_queue.push(ep_id);
    visited.insert(ep_id);
    sim_cache[ep_id] = ep_sim;

    // BFS扩展
    while (!bfs_queue.empty()) {
        uint64_t current_node = bfs_queue.front();
        bfs_queue.pop();

        // 遍历邻居节点
        for (uint64_t neighbor : nodes[current_node].neighbor[level]) {
            if (visited.count(neighbor)) continue;
            visited.insert(neighbor);

            // 检查邻居节点是否存在
            if (store.vectorStore.find(nodes[neighbor].key) == store.vectorStore.end()) continue;

            // 计算相似度（优先从缓存中读取）
            float sim;
            if (sim_cache.count(neighbor)) {
                sim = sim_cache[neighbor];
            } else {
                sim = cosineSimilarity(store.vectorStore[nodes[neighbor].key], query_vec);
                sim_cache[neighbor] = sim;
            }

            // 更新候选队列
            if ((int)top_candidates.size() < ef || sim > top_candidates.top().first) {
                top_candidates.emplace(sim, neighbor);
                if ((int)top_candidates.size() > ef) {
                    top_candidates.pop();
                }
                bfs_queue.push(neighbor);
            }
        }
    }

    // 提取结果
    std::vector<uint64_t> result;
    while (!top_candidates.empty()) {
        result.push_back(top_candidates.top().second);
        top_candidates.pop();
    }
    std::reverse(result.begin(), result.end()); // 按相似度从高到低排序
    return result;
}

// // 删除节点的函数
// 从 HNSW 图中移除该节点
// if (nodes.count(key)) {
//     for (auto &[level, neighbors] : nodes[key].neighbor) {
//         for (uint64_t neighbor : neighbors) {
//             auto &nb_list = nodes[neighbor].neighbor[level];
//             nb_list.erase(std::remove(nb_list.begin(), nb_list.end(), key), nb_list.end());
//         }
//     }
//     nodes.erase(key);
// }
//
// // 特殊处理：若 entrypoint 就是这个 key，需要重新设置 entrypoint
// if (entry_point == key) {
//     entry_point = UINT64_MAX;
//     for (const auto &pair : nodes) {
//         entry_point = pair.first;
//         break;
//     }
// }