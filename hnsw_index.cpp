
#include "hnsw_index.h"
#include "kvstore.h"
#include <cstdlib>
#include <cmath>
#include <limits>
#include <queue>
#include <algorithm>

HNSWIndex::HNSWIndex() {}
HNSWIndex::~HNSWIndex() {
    nodes.clear();
}

void HNSWIndex::del(uint64_t key) {
    // 从 HNSW 图中移除该节点
    if (nodes.count(key)) {
        for (auto &[level, neighbors] : nodes[key].neighbor) {
            for (uint64_t neighbor : neighbors) {
                auto &nb_list = nodes[neighbor].neighbor[level];
                nb_list.erase(std::remove(nb_list.begin(), nb_list.end(), key), nb_list.end());
            }
        }
        nodes.erase(key);
    }

    // 特殊处理：若 entrypoint 就是这个 key，需要重新设置 entrypoint
    if (entry_point == key) {
        entry_point = UINT64_MAX;
        for (const auto &pair : nodes) {
            entry_point = pair.first;
            break;
        }
    }
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
    if (nodes.find(key) == nodes.end()) {
        nodes[key] = HNSWNode{key, {}};
    }
    if (entry_point == UINT64_MAX) { // First node
        entry_point = key;
        max_level = level;
        return;
    }

     // if not the first node
     uint64_t ep = entry_point;
      for (int l = max_level; l > level; --l) {
          auto ep_neighbors = search_layer(store, ep, vec, l, 1);
          if (!ep_neighbors.empty()) ep = ep_neighbors[0];
      }
     //uint64_t ep = searchLayersGreedy(entry_point, vec, max_level, level + 1);

     for (int l = std::min(level, max_level); l >= 0; --l) {
         auto neighbors = search_layer(store, ep, vec, l, efConstruction);
         std::vector<std::pair<float, uint64_t>> scored;
         for (uint64_t nid : neighbors)
             scored.emplace_back(cosineSimilarity(vec, store.vectorStore[nid]), nid);
         std::partial_sort(scored.begin(), scored.begin() + std::min(M, (int)scored.size()), scored.end(), std::greater<>());

         std::vector<uint64_t> selected;
         selected.reserve(std::min(M, (int)scored.size()));
         for (int i = 0; i < std::min(M, (int)scored.size()); ++i)
             selected.push_back(scored[i].second);

         for (uint64_t neighbor : selected) {
             nodes[neighbor].neighbor[l].push_back(key);
             nodes[key].neighbor[l].push_back(neighbor);

             if (nodes[neighbor].neighbor[l].size()> M_max) {
                 auto &neigh_vecs = nodes[neighbor].neighbor[l];
                 std::vector<std::pair<float, uint64_t>> dist_list;
                 for (uint64_t nid : neigh_vecs)
                     dist_list.emplace_back(cosineSimilarity(store.vectorStore[nid], store.vectorStore[neighbor]), nid);
                 std::partial_sort(dist_list.begin(), dist_list.begin() + M_max, dist_list.end(), std::greater<>());
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
    if (store.vectorStore.find(ep_id) == store.vectorStore.end()) return {};

    // 初始化
    float ep_sim = cosineSimilarity(store.vectorStore[ep_id], query_vec);
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
            if (store.vectorStore.find(neighbor) == store.vectorStore.end()) continue;

            // 计算相似度（优先从缓存中读取）
            float sim;
            if (sim_cache.count(neighbor)) {
                sim = sim_cache[neighbor];
            } else {
                sim = cosineSimilarity(store.vectorStore[neighbor], query_vec);
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
