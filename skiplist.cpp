#include "skiplist.h"
#include <vector>


double skiplist::my_rand() {
    return (double)rand() / RAND_MAX;
}

int skiplist::randLevel() {
    int ret = 1;
    while (my_rand() < this->p && ret < MAX_LEVEL)
        ++ret;
    return ret;
}

//插入一个key-value对，若key已存在，则修改value，否则插入新的key-value对，若超出原有行数，则增加新的行
void skiplist::insert(uint64_t key, const std::string &str) {
    slnode *cur = this->head;
    std::vector<slnode *> update(MAX_LEVEL);
    int tmp = this->curMaxL;
    while (true) {
        while (cur->nxt[tmp] && cur->nxt[tmp]->type != TAIL && cur->nxt[tmp]->key < key)
            cur = cur->nxt[tmp];
        update[tmp] = cur;
        if (tmp == 1)
            break;
        --tmp;
    }

    //如果该key已存在，则仅修改该key的值，先判断有几层，对每层都要修改
    if (cur->nxt[1] && cur->nxt[1]->key == key) {
        bytes += str.length() - cur->nxt[1]->val.length();
        for (int i = 1; i <= this->curMaxL; ++i)
            if (update[i]->nxt[i] && update[i]->nxt[i] == cur->nxt[1])
                update[i]->nxt[i]->val = str;
            else
                break;
        return;
    }

    //否则，插入新的key
    int level = randLevel();
    slnode *new_node = new slnode(key, str, NORMAL);

    for (int i = 1; i <= this->curMaxL; ++i) { //若没有超出原有行数
        auto left = update[i];
        new_node->nxt[i] = left->nxt[i];
        left->nxt[i] = new_node;
        if (i >= level)
            break;
    }

    if (level <= this->curMaxL){
        bytes += sizeof(key) + str.length();
        return;
    }

    for (int i = this->curMaxL + 1; i <= level; ++i) { //若超出原有行数
        new_node->nxt[i] = this->tail;
        this->head->nxt[i] = new_node;
        this->curMaxL = level;
    }
    bytes += sizeof(key) + str.length();
}

//查找key对应的value，若不存在则返回空串
std::string skiplist::search(uint64_t key) {
    auto cur = this->head;
    for (int i = this->curMaxL; i >= 1; --i)
    {
        while (cur->nxt[i] && cur->nxt[i]->type != TAIL && cur->nxt[i]->key < key)
            cur = cur->nxt[i];
    }
    if (cur->nxt[1] && cur->nxt[1]->key == key)
        return cur->nxt[1]->val;
    return "";
}

//删除key-value对，若key不存在则返回false，否则返回true
bool skiplist::del(uint64_t key, uint32_t len)
{
    auto cur = this->head;
    std::vector<slnode *> update(MAX_LEVEL);
    for (int i = this->curMaxL; i >= 1; --i)
    {
        while (cur->nxt[i] && cur->nxt[i]->type != TAIL && cur->nxt[i]->key < key)
            cur = cur->nxt[i];
        update[i] = cur;
    }
    if (!cur->nxt[1] || cur->nxt[1]->key != key)
        return false;
    cur = cur->nxt[1];
    len = cur->val.length();
    for (int i = 1; i <= this->curMaxL; ++i)
    {
        if (update[i]->nxt[i] != cur)
            break;
        update[i]->nxt[i] = cur->nxt[i];
    }
    delete cur;
    while (this->curMaxL > 1 && this->head->nxt[this->curMaxL] == this->tail)
        --this->curMaxL;
    this->bytes -= (sizeof(key) + len);
    return true;
}

//查找key1到key2之间的所有key-value对，返回一个vector，其中每个元素为一个pair，first为key，second为value
void skiplist::scan(uint64_t key1, uint64_t key2, std::vector<std::pair<uint64_t, std::string>> &list) {
    auto cur = this->head;
    for (int i = this->curMaxL; i >= 1; --i)
    {
        while (cur->nxt[i] && cur->nxt[i]->type != TAIL && cur->nxt[i]->key < key1)
            cur = cur->nxt[i];
    }
    if (cur->nxt[1])
        cur = cur->nxt[1];
    while (cur && cur->type != TAIL && cur->key <= key2)
    {
        list.emplace_back(cur->key, cur->val);
        cur = cur->nxt[1];
    }
}

/*返回大于等于的第一个 没有返回len + 1*/
slnode *skiplist::lowerBound(uint64_t key) {
    auto cur = this->head;
    for (int i = this->curMaxL; i >= 1; --i)
    {
        while (cur->nxt[i]->type != TAIL && cur->nxt[i]->key < key)
            cur = cur->nxt[i];
    }
    return cur->nxt[1] ? cur->nxt[1] : this->tail;
}

//重置跳表
void skiplist::reset() {
    auto cur = this->head->nxt[1];
    while (cur && cur->type != TAIL)
    {
        auto tmp = cur;
        cur = cur->nxt[1];
        delete tmp;
    }
    for (int i = 1; i <= this->curMaxL; ++i)
        this->head->nxt[i] = this->tail;
    this->curMaxL = 1;
    this->bytes = 0;
}

uint32_t skiplist::getBytes()
{
    return this->bytes;
}


skiplist::~skiplist()
{
    reset();
    delete this->head;
    delete this->tail;
}