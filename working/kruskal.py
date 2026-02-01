from itertools import combinations

def kruskal(n, edges):
    # edges = [(w,u,v), ...] where u,v in [0..n-1]
    parent = list(range(n))
    size = [1] * n

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return False
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
        return True

    mst = []
    cost = 0.0
    for w, u, v in sorted(edges):
        if union(u, v):
            mst.append((u, v, w))
            cost += w
            if len(mst) == n - 1:
                break
    return mst, cost


def best_mst_over_all_16_node_subsets(nodes, edges, k=16):
    """
    nodes: list of node IDs (length 18)
    edges: list of (w, u, v) with u,v being node IDs from `nodes`
    Returns: (best_subset, best_mst_edges, best_cost)
    where best_mst_edges are (u,v,w) in ORIGINAL node IDs.
    """
    best_cost = float("inf")
    best_subset = None
    best_mst = None

    for subset in combinations(nodes, k):
        S = set(subset)
        # keep only edges fully inside the subset
        sub_edges = [(w, u, v) for (w, u, v) in edges if u in S and v in S]

        # remap node IDs in subset -> 0..k-1 for DSU
        idx = {node: i for i, node in enumerate(subset)}
        sub_edges_idx = [(w, idx[u], idx[v]) for (w, u, v) in sub_edges]

        mst_idx, cost = kruskal(k, sub_edges_idx)

        # if not connected, skip (MST would have < k-1 edges)
        if len(mst_idx) != k - 1:
            continue

        if cost < best_cost:
            best_cost = cost
            best_subset = subset
            # map MST edges back to original node IDs
            inv = list(subset)  # inv[i] is original node id
            best_mst = [(inv[u], inv[v], w) for (u, v, w) in mst_idx]

    return best_subset, best_mst, best_cost
# raw data
data = [([18, 17], 0.18368804661031968), ([9, 10], 0.29178665557484385), ([13, 12], 1.1162589543305956), ([18, 19], 0.12760287712831886), ([11, 10], 0.4849068709610771), ([13, 14], 0.9584088942435565), ([1, 4], 0.21131905149636143), ([13, 8], 0.4934355869787632), ([15, 10], 0.5757440802106051), ([9, 4], 0.5104884287033373), ([11, 6], 0.25524898234603466), ([13, 17], 0.42224936140939917), ([15, 19], 0.36463650336719944), ([7, 12], 0.6137507088124439), ([9, 14], 0.6632456296422617), ([11, 16], 0.39494925123836344), ([5, 10], 0.8317877542178387), ([18, 14], 0.39979274222825545), ([5, 4], 0.8913503940611966), ([7, 8], 1.2429394607743616), ([15, 14], 0.26519635616050197), ([1, 0], 0.2398695990178834), ([5, 6], 0.5416455042012602), ([9, 8], 1.0088600974843764), ([15, 16], 0.38932518390986104)]
nodes = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]  # your 18 nodes
edges = [(w, i, j) for ([i, j], w) in data]

subset, mst_edges, cost = best_mst_over_all_16_node_subsets(nodes, edges, k=16)

print("Best subset:", subset)
print("Best cost:", cost)
print("MST edges:", [[u,v] for (u,v,_) in mst_edges])
print("Num MST edges:", len(mst_edges))  # should be 15