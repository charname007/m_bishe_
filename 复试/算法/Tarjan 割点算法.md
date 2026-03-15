在图论中，**割点（Articulation Point / Cut Vertex）**是指在一个连通图中，如果删除该点及其相关的边，图的连通分量就会增加（即图会变得不连通）。

Tarjan 算法是基于 **DFS（深度优先遍历）** 的高效算法，其核心思想是判断一个点是否能通过其子树中的节点“绕路”回溯到该点之上的祖先节点。

---

## 1. 核心概念：$dfn$ 与 $low$

理解 Tarjan 的关键在于维护两个数组：

- **$dfn[u]$（时间戳）：** 节点 $u$ 在 DFS 过程中被访问的次序。
    
- **$low[u]$（追溯值）：** 在 DFS 树中，$u$ 或其子树能够通过**反向边（Back Edge）**到达的 $dfn$ 最小的节点的时间戳。
    

---

## 2. 割点的判定法则

假设我们在 DFS 树中正在处理节点 $u$，并有一个子节点 $v$：

## 情况 A：$u$ 是 DFS 树的根节点

如果 $u$ 是根，且它在 DFS 树中有 **两个及以上** 的子树（注意：是 DFS 树中的子节点，不是度数），那么 $u$ 就是割点。

- _直观理解：_ 根节点连着两坨互不相通的子树，删了根，两坨就散了。
    

## 情况 B：$u$ 不是根节点

如果存在一个子节点 $v$，满足：

$$low[v] \geq dfn[u]$$

那么 $u$ 就是割点。

- _直观理解：_ 这意味着 $v$ 及其子树里的所有点，无论怎么绕，最高也只能绕到 $u$。一旦 $u$ 消失，$v$ 这一脉就彻底和 $u$ 以上的部分断开了。
    

---

## 3. 算法实现步骤 (C++)

我们可以利用 C++ 的递归很方便地实现。注意，割点判定时要记得去重，因为一个点可能满足多个子节点的判定条件。

C++

```
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

const int MAXN = 100005;
vector<int> adj[MAXN];
int dfn[MAXN], low[MAXN], timer;
bool is_cut[MAXN];

void tarjan(int u, int p = -1) {
    dfn[u] = low[u] = ++timer;
    int children = 0;

    for (int v : adj[u]) {
        if (v == p) continue; // 避免直接跑回父亲
        if (dfn[v]) {
            // v 已经访问过，说明 (u, v) 是一条反向边
            low[u] = min(low[u], dfn[v]);
        } else {
            // v 是新节点，递归搜索
            children++;
            tarjan(v, u);
            low[u] = min(low[u], low[v]);
            
            // 判定准则：非根节点且满足 low[v] >= dfn[u]
            if (p != -1 && low[v] >= dfn[u])
                is_cut[u] = true;
        }
    }
    // 特判根节点
    if (p == -1 && children > 1)
        is_cut[u] = true;
}
```

---

## 4. 关键点拨

1. **关于 $low[u] = \min(low[u], dfn[v])$：** 在遇到已访问节点（反向边）时，只能用 $dfn[v]$ 更新，而不能用 $low[v]$。虽然在求割点时有时混用能过，但在求强连通分量（SCC）或其他复杂变体时，混用会导致逻辑错误。
    
2. **时间复杂度：** $O(V + E)$，每个点和每条边都只被访问常数次，非常高效。
    

---

**想试试手吗？** 如果你愿意，我可以出一道关于图连通性的练习题，或者帮你分析一下这个算法在 **双连通分量（BCC）** 中的进一步应用。