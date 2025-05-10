# Smart LSM-tree with Persistent HNSW
===
本项目基于传统的 LSM-tree 构建了一个支持高效向量检索的键值存储系统，并集成了 HNSW（Hierarchical Navigable Small World） 图结构，实现了向量索引的持久化、Lazy Delete、修改操作等功能，支持端到端的语义搜索。

# 项目结构：
```
.
├── src/                  # 项目源代码（KVStore、HNSWIndex等核心组件）
├── data/                 # 存储 KV 数据、SSTables、embedding 文件
├── hnsw_data/            # HNSW 索引结构持久化目录
├── test/                 # 实验与测试脚本
│   ├── Embedding_Test.cpp
│   ├── HNSW_KNN_Test.cpp
│   ├── HNSW_Persistent_Test_Phase1.cpp
│   └── HNSW_Persistent_Test_Phase2.cpp
└── README.md
```

Phase 1: 插入、删除、持久化索引
./build/test/HNSW_Persistent_Test_Phase1

Phase 2: 加载索引，验证删除/重新插入/替换的正确性
./build/test/HNSW_Persistent_Test_Phase2

# 已完成功能
 1. 支持嵌入向量生成与存储（append 模式）

 2. 向量检索接口 search_knn_hnsw 基于 HNSW 实现

 3. 支持节点删除（Lazy Delete）与重新插入

 4. HNSW 索引结构持久化到磁盘

 5. 系统重启后从磁盘恢复向量与索引数据

 6. 自动跳过已删除节点参与查询

 7. 单元测试覆盖插入、删除、替换等典型操作

# 数据文件说明
embedding_data/embedding_vectors.bin：所有向量的二进制拼接文件（每个 entry 为 8B key + dim×4B 向量）

hnsw_data/：持久化的 HNSW 索引目录，包含：

global_header.bin：HNSW 全局参数

deleted_nodes.bin：标记为已删除的节点 ID

nodes/：每个节点一个子目录，保存其 key 与邻接边数据

# 实现
Embedding 向量持久化：采用 append-only 二进制文件，文件头存储维度信息，每条记录为 (key, vector)，启动时倒序扫描取最新有效项。

HNSW 索引持久化：为每个节点分配自增 ID，并以 ID 为目录名存储其 key 与每层邻接边。

删除与修改支持：使用 unordered_set 记录已删除 ID，查询时跳过；插入同一个 key 会标记旧 ID 为 deleted。

# 致谢
本项目参考了课程实验框架和 HNSW 原始论文，感谢课程组提供的测试样例与评测代码支持。
