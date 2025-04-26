#include "kvstore.h"

#include "skiplist.h"
#include "sstable.h"
#include "utils.h"
#include "embedding/embedding.h"
#include <cmath>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <unordered_set>

static const std::string DEL = "~DELETED~";
const uint32_t MAXSIZE = 2 * 1024 * 1024;
double put_embedding_time = 0.0;
double search_embedding_time = 0.0;

struct poi {
    int sstableId; // vector中第几个sstable
    int pos; // 该sstable的第几个key-offset
    uint64_t time;
    Index index;
};

struct cmpPoi {
    bool operator()(const poi &a, const poi &b) {
        if (a.index.key == b.index.key)
            return a.time < b.time;
        return a.index.key > b.index.key;
    }
};

KVStore::KVStore(const std::string &dir) : KVStoreAPI(dir) // read from sstables
{
    for (totalLevel = 0;; ++totalLevel) {
        std::string path = dir + "/level-" + std::to_string(totalLevel) + "/";
        std::vector<std::string> files;
        if (!utils::dirExists(path)) {
            totalLevel--;
            break; // stop read
        }
        int nums = utils::scanDir(path, files);
        sstablehead cur;
        for (int i = 0; i < nums; ++i) {
            // 读每一个文件头
            std::string url = path + files[i]; // url, 每一个文件名
            cur.loadFileHead(url.data());
            sstableIndex[totalLevel].push_back(cur);
            TIME = std::max(TIME, cur.getTime()); // 更新时间戳
        }
    }
    loadVectorsFromSSTables();
}
void KVStore::loadVectorsFromSSTables() {
    for (int level = 0; level <= totalLevel; ++level) {
        for (sstablehead it : sstableIndex[level]) {
            std::string url = it.getFilename();
            sstable ss;
            ss.loadFile(url.data());
            for (size_t i = 0; i < ss.getCnt(); ++i) {
                uint64_t key = ss.getKey(i);
                std::string value = ss.getData(i);
                std::vector<float> vector = embedding_single(value);
                if (vector.empty()) {
                    std::cerr << "Failed to embed the value" << std::endl;
                    continue;
                }
                vectorStore[key] = vector;
                insert_hnsw_node(key, vector);
            }
        }
    }
}
void KVStore::insert_hnsw_node(const std::uint64_t& key, const std::vector<float>& vec) {
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
          auto ep_neighbors = search_layer(ep, vec, l, 1);
          if (!ep_neighbors.empty()) ep = ep_neighbors[0];
      }
     //uint64_t ep = searchLayersGreedy(entry_point, vec, max_level, level + 1);

     for (int l = std::min(level, max_level); l >= 0; --l) {
         auto neighbors = search_layer(ep, vec, l, efConstruction);
         std::vector<std::pair<float, uint64_t>> scored;
         for (uint64_t nid : neighbors)
             scored.emplace_back(cosineSimilarity(vec, vectorStore[nid]), nid);
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
                     dist_list.emplace_back(cosineSimilarity(vectorStore[nid], vectorStore[neighbor]), nid);
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


KVStore::~KVStore() {
    sstable ss(s);
    if (!ss.getCnt())
        return; // empty sstable
    std::string path = std::string("./data/level-0/");
    if (!utils::dirExists(path)) {
        utils::_mkdir(path.data());
        totalLevel = 0;
    }
    ss.putFile(ss.getFilename().data());
    compaction(); // 从0层开始尝试合并
    vectorStore.clear();
    nodes.clear();
}

/**
 * Insert/Update the key-value pair.
 * No return values for simplicity.
 */
void KVStore::put(uint64_t key, const std::string &val) {
    uint32_t nxtsize = s->getBytes();
    std::string res = s->search(key);
    if (!res.length()) {
        // new add
        nxtsize += 12 + val.length();
    } else
        nxtsize = nxtsize - res.length() + val.length(); // change string
    if (nxtsize + 10240 + 32 <= MAXSIZE)
        s->insert(key, val); // 小于等于（不超过） 2MB
    else {
        sstable ss(s);
        s->reset();
        std::string url = ss.getFilename();
        std::string path = "./data/level-0";
        if (!utils::dirExists(path)) {
            utils::mkdir(path.data());
            totalLevel = 0;
        }
        addsstable(ss, 0); // 加入缓存
        ss.putFile(url.data()); // 加入磁盘
        compaction();
        s->insert(key, val);
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> vector = embedding_single(val);
    auto end = std::chrono::high_resolution_clock::now();
    put_embedding_time += std::chrono::duration<double>(end - start).count();
    vectorStore[key] = vector;
    insert_hnsw_node(key, vector);
}

/**
 * Returns the (string) value of the given key.
 * An empty string indicates not found.
 */
std::string KVStore::get(uint64_t key) //
{
    uint64_t time = 0;
    int goalOffset;
    uint32_t goalLen;
    std::string goalUrl;
    std::string res = s->search(key);
    if (res.length()) {
        // 在memtable中找到, 或者是deleted，说明最近被删除过，
        // 不用查sstable
        if (res == DEL)
            return "";
        return res;
    }
    for (int level = 0; level <= totalLevel; ++level) {
        for (sstablehead it: sstableIndex[level]) {
            if (key < it.getMinV() || key > it.getMaxV())
                continue;
            uint32_t len;
            int offset = it.searchOffset(key, len);
            if (offset == -1) {
                if (!level)
                    continue;
                else
                    break;
            }
            // sstable ss;
            // ss.loadFile(it.getFilename().data());
            if (it.getTime() > time) {
                // find the latest head
                time = it.getTime();
                goalUrl = it.getFilename();
                goalOffset = offset + 32 + 10240 + 12 * it.getCnt();
                goalLen = len;
            }
        }
        if (time)
            break; // only a test for found
    }
    if (!goalUrl.length())
        return ""; // not found a sstable
    res = fetchString(goalUrl, goalOffset, goalLen);
    if (res == DEL)
        return "";
    return res;
}

/**
 * Delete the given key-value pair if it exists.
 * Returns false iff the key is not found.
 */
bool KVStore::del(uint64_t key) {
    std::string res = get(key);
    if (!res.length())
        return false; // not exist
    put(key, DEL); // put a del marker
    vectorStore.erase(key);

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

    return true;
}

/**
 * This resets the kvstore. All key-value pairs should be removed,
 * including memtable and all sstables files.
 */
void KVStore::reset() {
    s->reset(); // 先清空memtable
    std::vector<std::string> files;
    for (int level = 0; level <= totalLevel; ++level) {
        // 依层清空每一层的sstables
        std::string path = std::string("./data/level-") + std::to_string(level);
        int size = utils::scanDir(path, files);
        for (int i = 0; i < size; ++i) {
            std::string file = path + "/" + files[i];
            utils::rmfile(file.data());
        }
        utils::rmdir(path.data());
        sstableIndex[level].clear();
    }
    totalLevel = -1;
    vectorStore.clear();
    nodes.clear();
    entry_point = UINT64_MAX;
    max_level = -1;
}

/**
 * Return a list including all the key-value pair between key1 and key2.
 * keys in the list should be in an ascending order.
 * An empty string indicates not found.
 */

struct myPair {
    uint64_t key, time;
    int id, index;
    std::string filename;

    myPair(uint64_t key, uint64_t time, int index, int id,
           std::string file) {
        // construct function
        this->time = time;
        this->key = key;
        this->id = id;
        this->index = index;
        this->filename = file;
    }
};

struct cmp {
    bool operator()(myPair &a, myPair &b) {
        if (a.key == b.key)
            return a.time < b.time;
        return a.key > b.key;
    }
};


void KVStore::scan(uint64_t key1, uint64_t key2, std::list<std::pair<uint64_t, std::string> > &list) {
    std::vector<std::pair<uint64_t, std::string> > mem;
    // std::set<myPair> heap; // 维护一个指针最小堆
    std::priority_queue<myPair, std::vector<myPair>, cmp> heap;
    // std::vector<sstable> ssts;
    std::vector<sstablehead> sshs;
    s->scan(key1, key2, mem); // add in mem
    std::vector<int> head, end; // [head, end)
    int cnt = 0;
    if (mem.size())
        heap.push(myPair(mem[0].first, INF, 0, -1, "qwq"));
    for (int level = 0; level <= totalLevel; ++level) {
        for (sstablehead it: sstableIndex[level]) {
            if (key1 > it.getMaxV() || key2 < it.getMinV())
                continue; // 无交集
            int hIndex = it.lowerBound(key1);
            int tIndex = it.lowerBound(key2);
            if (hIndex < it.getCnt()) {
                // 此sstable可用
                // sstable ss; // 读sstable
                std::string url = it.getFilename();
                // ss.loadFile(url.data());

                heap.push(myPair(it.getKey(hIndex), it.getTime(), hIndex, cnt++, url));
                head.push_back(hIndex);
                if (it.search(key2) == tIndex)
                    tIndex++; // tIndex为第一个不可的
                end.push_back(tIndex);
                // ssts.push_back(ss); // 加入ss
                sshs.push_back(it);
            }
        }
    }
    uint64_t lastKey = INF; // only choose the latest key
    while (!heap.empty()) {
        // 维护堆
        myPair cur = heap.top();
        heap.pop();
        if (cur.id >= 0) {
            // from sst
            if (cur.key != lastKey) {
                lastKey = cur.key;
                uint32_t start = sshs[cur.id].getOffset(cur.index - 1);
                uint32_t len = sshs[cur.id].getOffset(cur.index) - start;
                uint32_t scnt = sshs[cur.id].getCnt();
                std::string res = fetchString(cur.filename, 10240 + 32 + scnt * 12 + start, len);
                if (res.length() && res != DEL)
                    list.emplace_back(cur.key, res);
            }
            if (cur.index + 1 < end[cur.id]) {
                // add next one to heap
                heap.push(myPair(sshs[cur.id].getKey(cur.index + 1), cur.time, cur.index + 1, cur.id, cur.filename));
            }
        } else {
            // from mem
            if (cur.key != lastKey) {
                lastKey = cur.key;
                std::string res = mem[cur.index].second;
                if (res.length() && res != DEL)
                    list.emplace_back(cur.key, mem[cur.index].second);
            }
            if (cur.index < mem.size() - 1) {
                heap.push(myPair(mem[cur.index + 1].first, cur.time, cur.index + 1, -1, cur.filename));
            }
        }
    }
}

/*
若Levcl 0层中的文件数量超出限制，则开始进行合并操作。对于Level0层的合并操作来说，需要将所有的Level0层中的SSTable 与 Level1 层的中部分
SSTable 进行合并，随后将产生的新SSTable 文件写入到Ievel 1层中：
1.先统计Level0层中所有SSTable 所覆盖的键的区间。然后在 Level 1层中找到与此区间有交集的所有 SSTable 文件。
2.使用归并排序，将上述所有涉及到的 SSTable 进行合并，并将结果每2MB 分成一个新的 SSTable 文件（最后一个 SSTable 可以不足2MB），写入到 Level1中。
3. 若产生的文件数超出Level 1层限定的数目，则从level 1的SSTable
中，优先选择时间戳最小的若干个文件（时间戳相等选择键最小的文件），使得文件数满足层数要求，以同样的方法继续向下一层合并（若没有下一层，则新建一层）。
注意
1）从Level1层往下的合并开始，仅需将超出的文件往下一层进行合并即可，无需合并该层所有文件。
2）在合并时，如果遇到相同键K的多条记录，通过比较时间戳来决定键K的最新值，时间戳大的记录被保留。最后一层的合并需要将标为deleted的记录删除。
3）完成一次合并操作之后需要更新涉及到的SSTable 在内存中的缓存信息
4）每层最多pow(2, level + 1)个文件
 */
void KVStore::compaction() {
    if (totalLevel == -1) totalLevel = totalLevel + 1;
    if (sstableIndex[0].size() <= pow(2, 1)) return;

    int level = 0;
    while (true) {
        // 0. 递归终止条件
        if (level > totalLevel || sstableIndex[level].empty()) break;

        // 1. 选取 level 中超出限度的SSTable 和 Level+1 中有交集的 SSTable
        std::vector<sstablehead> involvedTables;
        if (level == 0) {
            //involvedTables = std::move(sstableIndex[level]);
            involvedTables = sstableIndex[level];
        } else {
            int overLimit = sstableIndex[level].size() - pow(2, level + 1);
            if (overLimit > 0) {
                std::vector<sstablehead> tempTables = sstableIndex[level];
                std::sort(tempTables.begin(), tempTables.end());
                involvedTables.assign(tempTables.begin(), tempTables.begin() + overLimit);
            }
        }

        // 2. 计算 Level 的 Key 范围
        uint64_t minKey = UINT64_MAX, maxKey = 0;
        for (const auto &sst: involvedTables) {
            minKey = std::min(minKey, sst.getMinV());
            maxKey = std::max(maxKey, sst.getMaxV());
        }

        if (level + 1 <= totalLevel) {
            for (sstablehead &it: sstableIndex[level + 1]) {
                if (it.getMaxV() >= minKey && it.getMinV() <= maxKey) {
                    involvedTables.push_back(it);
                }
            }
        }

        // 3. 归并排序
        std::priority_queue<myPair, std::vector<myPair>, cmp> pq; // key大的、time大的排后面
        std::unordered_map<int, std::vector<std::pair<uint64_t, std::string> > > sstData;
        int sst_id = 0;
        for (const auto &sst: involvedTables) {
            std::string filename = sst.getFilename();
            if (filename.empty()) continue; // 避免 loadFile 崩溃
            sstable temp;
            temp.loadFile(filename.c_str());
            sstData[sst_id].reserve(temp.getCnt());
            for (size_t i = 0; i < temp.getCnt(); ++i) {
                uint64_t key = temp.getKey(i);
                std::string value = temp.getData(i);
                pq.push(myPair(key, temp.getTime(), i, sst_id, sst.getFilename()));
                sstData[sst_id].emplace_back(key, value);
            }
            ++sst_id;
        }

        uint64_t max_timestamp = 0;
        for (const auto &sst : involvedTables) {
            max_timestamp = std::max(max_timestamp, sst.getTime());  // 取最大时间戳
        }
        uint64_t prev_TIME = TIME;
        TIME = max_timestamp;

        // 4. 生成最新的键值对
        std::unordered_map<uint64_t, std::string> latestData;
        while (!pq.empty()) {
            myPair top = pq.top();
            pq.pop();

            uint64_t key = top.key;
            int sst_id = top.id;
            size_t index = top.index;
            uint64_t time = top.time;

            // 仅保留 time 最大的数据
            if (latestData.find(key) == latestData.end() || time > pq.top().time) {
                latestData[key] = sstData[sst_id][index].second;
            }
        }

        // 5. 在最底层删除被标记为 deleted 的键
        if (level == totalLevel) {
            // 只有最底层才删除
            std::vector<uint64_t> deleteKeys;

            // 1. 找出所有 value == DEL 的 key
            for (const auto &entry: latestData) {
                if (entry.second == DEL) {
                    deleteKeys.push_back(entry.first);
                }
            }

            // 2. 遍历 latestData，删除所有出现在 deleteKeys 里的 key
            for (auto it = latestData.begin(); it != latestData.end();) {
                bool toDelete = false;
                for (const auto &key: deleteKeys) {
                    if (it->first == key) {
                        // 不能用 find，直接比较
                        toDelete = true;
                        break;
                    }
                }
                if (toDelete) {
                    it = latestData.erase(it);
                } else {
                    ++it;
                }
            }
        }

        std::vector<std::pair<uint64_t, std::string> > sortedData(latestData.begin(), latestData.end());
        std::sort(sortedData.begin(), sortedData.end());

        // 6. 拆分新 SSTable 并存入 level + 1
        sstable newSST;


        // 如果没有下级，创建下一层目录，总层数加一；如果有下级，获取下一层目录。
        std::string nextLevelDir = "./data/level-" + std::to_string(level + 1);
        if (!utils::dirExists(nextLevelDir)) {
            totalLevel = level + 1;
            utils::mkdir(nextLevelDir.c_str());
        }
        //nameSuffix

        int maxNameSuffix = -1;
        std::vector<std::string> filenames;
        utils::scanDir(nextLevelDir, filenames);  // 获取目录下所有文件名

        for (const auto& fname : filenames) {
            size_t dashPos = fname.find('-');
            size_t dotPos = fname.find(".sst");
            if (dashPos != std::string::npos && dotPos != std::string::npos) {
                uint64_t fileTime = std::stoull(fname.substr(0, dashPos));
                if (fileTime == TIME) {  // 时间戳匹配
                    int suffix = std::stoi(fname.substr(dashPos + 1, dotPos - dashPos - 1));
                    maxNameSuffix = std::max(maxNameSuffix, suffix);
                }
            }
        }
        newSST.setTime(TIME);
        newSST.setNamesuffix(maxNameSuffix + 1);

        for (const auto &[key, value]: sortedData) {
            if (newSST.checkSize(value, level + 1, 0)) {
                newSST.setTime(TIME);
                newSST.putFile(newSST.getFilename().c_str());
                addsstable(newSST, level + 1);
                newSST.reset();
            }
            newSST.insert(key, value);
        }
        if (newSST.getCnt() > 0) {
            std::string value = "";
            newSST.checkSize(value, level + 1, 1);
            newSST.setTime(TIME);
            newSST.putFile(newSST.getFilename().c_str());
            addsstable(newSST, level + 1);
            newSST.reset();
        }

        TIME = prev_TIME;

        // 7. 删除旧 SSTable
        for (const auto &sst: involvedTables) {
            if (access(sst.getFilename().c_str(), F_OK) == 0)
                delsstable(sst.getFilename());
        }

        for (const auto &sst: involvedTables) {
            utils::rmfile(sst.getFilename().c_str());
        }

        involvedTables.clear();


        // 8. 检查 level + 1 是否超限
        if (level + 1 <= totalLevel && sstableIndex[level + 1].size() > pow(2, level + 2)) {
            // 如果有下层且下层超限
            level++; // 递归到下一层
        } else {
            // 否则退出
            break;
        }
    }
}


void KVStore::delsstable(std::string filename) {
    for (int level = 0; level <= totalLevel; ++level) {
        int size = sstableIndex[level].size(), flag = 0;
        for (int i = 0; i < size; ++i) {
            if (sstableIndex[level][i].getFilename() == filename) {
                sstableIndex[level].erase(sstableIndex[level].begin() + i);
                flag = 1;
                break;
            }
        }
        if (flag)
            break;
    }
    int flag = utils::rmfile(filename.data());
    if (flag != 0) {
        std::cout << "delete fail!" << std::endl;
        std::cout << strerror(errno) << std::endl;
    }
}

void KVStore::addsstable(sstable ss, int level) {
    sstableIndex[level].push_back(ss.getHead());
}

char strBuf[2097152];

/**
 * @brief Fetches a substring from a file starting at a given offset.
 *
 * This function opens a file in binary read mode, seeks to the specified start offset,
 * reads a specified number of bytes into a buffer, and returns the buffer as a string.
 *
 * @param file The path to the file from which to read the substring.
 * @param startOffset The offset in the file from which to start reading.
 * @param len The number of bytes to read from the file.
 * @return A string containing the read bytes.
 */
std::string KVStore::fetchString(std::string file, int startOffset, uint32_t len) {
    // TODO here
    FILE *fp = fopen(file.data(), "rb");
    if (!fp) {
        throw std::runtime_error("Failed to open file: " + file);
    }

    if (fseek(fp, startOffset, SEEK_SET) != 0) {
        fclose(fp);
        throw std::runtime_error("Failed to seek in file: " + file);
    }

    size_t bytesRead = fread(strBuf, 1, len, fp);
    fclose(fp);

    if (bytesRead != len) {
        throw std::runtime_error("Failed to read expected bytes from file: " + file);
    }

    return std::string(strBuf, strBuf + bytesRead);
}
//该接口接受一个查询字符串和一个整数 k，返回与查询字符串最相近的k个向量的key和value。并且按照向量余弦相似度从高到低的顺序排列。E2E_test.cpp不会因浮点数精度影响结果，之后的测试也会容忍一定的浮点数计算误差。
std::vector<std::pair<std::uint64_t, std::string>> KVStore::search_knn(std::string query, int k) {
    // 将查询字符串转化为向量
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> query_vector = embedding_single(query);
    auto end = std::chrono::high_resolution_clock::now();
    search_embedding_time += std::chrono::duration<double>(end - start).count();
    if (query_vector.empty()) {
        std::cerr << "Failed to embed the query" << std::endl;
        return {};
    }
    // 计算所有向量与查询向量的余弦相似度
    std::vector<std::pair<double, std::uint64_t>> similarities;
    similarities.reserve(vectorStore.size());

    for (const auto &entry : vectorStore) {
        uint64_t key = entry.first;
        const std::vector<float> &value_vector = entry.second;
        double similarity = cosineSimilarity(query_vector, value_vector);
        similarities.push_back({similarity, key});
    }
    // 按照余弦相似度从高到低排序
    std::sort(similarities.rbegin(), similarities.rend(),
        [](const std::pair<double, uint64_t> &a, const std::pair<double, uint64_t> &b) {
            return a.first < b.first;
        });

    // 取前 k 个
    std::vector<std::pair<std::uint64_t, std::string>> result;
    result.reserve(k);

    for (int i = 0; i < k && i < similarities.size(); ++i) {
        uint64_t key = similarities[i].second;
        std::string value = get(key);
        if (value != DEL) {
            result.push_back({key, value});
        }
    }
    return result;
}

float KVStore::cosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2) const {
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


std::vector<uint64_t> KVStore::search_layer(uint64_t ep_id, const std::vector<float> &query_vec, int level, int ef = efConstruction) {
    // ep_id为起始点，query_vec为要搜索的点的vector，level为当前搜索的层，efConstruction为维护当前层搜索到的与q最近邻的个数
    std::priority_queue<std::pair<float, uint64_t>, std::vector<std::pair<float, uint64_t>>, std::greater<>> top_candidates; // 最小堆
    std::queue<uint64_t> bfs_queue;
    std::unordered_set<uint64_t> visited;
    std::unordered_map<uint64_t, float> sim_cache; // 缓存相似度计算结果

    // 检查起始节点是否存在
    if (vectorStore.find(ep_id) == vectorStore.end()) return {};

    // 初始化
    float ep_sim = cosineSimilarity(vectorStore[ep_id], query_vec);
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
            if (vectorStore.find(neighbor) == vectorStore.end()) continue;

            // 计算相似度（优先从缓存中读取）
            float sim;
            if (sim_cache.count(neighbor)) {
                sim = sim_cache[neighbor];
            } else {
                sim = cosineSimilarity(vectorStore[neighbor], query_vec);
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

//该接口接受一个查询字符串和一个整数 k，返回与查询字符串最相近的k个向量的key和value。并且按照向量余弦相似度从高到低的顺序排列。E2E_test.cpp不会因浮点数精度影响结果，之后的测试也会容忍一定的浮点数计算误差。
std::vector<std::pair<std::uint64_t, std::string>> KVStore::search_knn_hnsw(std::string query, int k) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> query_vec = embedding_single(query);
    auto end = std::chrono::high_resolution_clock::now();
    search_embedding_time += std::chrono::duration<double>(end - start).count();

    if (query_vec.empty()) {
        std::cerr << "Failed to embed the query" << std::endl;
        return {};
    }

    uint64_t ep = entry_point;
    for (int l = max_level; l >= 1; --l) {
        auto next = search_layer(ep, query_vec, l, 1);
        if (!next.empty()) ep = next[0];
    }
    //uint64_t ep = searchLayersGreedy(entry_point, query_vec, max_level, 1);

    auto ef_results = search_layer(ep, query_vec, 0, efConstruction);
    std::vector<std::pair<float, uint64_t>> scored;
    for (uint64_t key : ef_results)
        scored.emplace_back(cosineSimilarity(query_vec, vectorStore[key]), key);

    std::partial_sort(scored.begin(), scored.begin() + std::min(k, (int)scored.size()), scored.end(), std::greater<>());

    std::vector<std::pair<std::uint64_t, std::string>> result;
    for (int i = 0; i < std::min(k, (int)scored.size()); ++i) {
        uint64_t key = scored[i].second;
        std::string value = get(key);
        if (value != DEL) {
            result.push_back({key, value});
        }
    }
    //不满 k 个的要补齐 k 个
    while (result.size() < k) {
        result.push_back({UINT64_MAX, ""}); // 填充一个无效的结果
    }
    return result;
}

uint64_t KVStore::searchLayersGreedy(uint64_t ep_id, const std::vector<float> &query_vec, int fromLevel, int toLevel) const {
    if (vectorStore.find(ep_id) == vectorStore.end()) {
        throw std::runtime_error("Entry point embedding not found in vectorStore.");
    }

    std::unordered_map<uint64_t, float> sim_cache; // 缓存相似度

    auto get_similarity = [&](uint64_t id) -> float {
        if (sim_cache.count(id)) return sim_cache[id];
        float sim = cosineSimilarity(vectorStore.at(id), query_vec);
        sim_cache[id] = sim;
        return sim;
    };

    uint64_t curr_node = ep_id;
    float curr_sim = get_similarity(curr_node);

    for (int level = fromLevel; level >= toLevel; --level) {
        bool changed = true;

        while (changed) {
            changed = false;

            // 当前节点的邻居
            const auto& neighbors_map = nodes.at(curr_node).neighbor;
            auto it = neighbors_map.find(level);
            if (it == neighbors_map.end()) break;

            for (uint64_t neighbor : it->second) {
                if (vectorStore.find(neighbor) == vectorStore.end()) continue;

                float sim = get_similarity(neighbor);
                if (sim > curr_sim) {
                    curr_node = neighbor;
                    curr_sim = sim;
                    changed = true;
                }
            }
        }
    }

    return curr_node;
}

