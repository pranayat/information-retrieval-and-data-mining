import numpy as np
import pandas as pd
import networkx as nx

import networkx as nx
import matplotlib.pyplot as plt

expansion_set = []
def HITS(path, rootIndex):
    a = range(1, 1001)
    b = range(1, 1001)
    a_m = pd.read_csv(path, sep=",", header=None)
    a_m.columns = a
    a_m.index = b

    a_m = np.array(a_m)
    i = 0
    j = 0

    for index, node in enumerate(a_m):
        # get all index where value is 1
        temp = [i + 1 for i in range(len(node)) if node[i] == 1]
        if index + 1 in rootIndex:
            for item in temp:
                expansion_set.insert(index + 1, (index + 1, item))
        else:
            check = any(item in temp for item in rootIndex)
            if check is True:
                for item in rootIndex:
                    if item in temp:
                        expansion_set.insert(index + 1, (index + 1, item))

    print(expansion_set)

    G = nx.DiGraph()
    G.add_edges_from(expansion_set)
    hubs, authorities = nx.hits(G, max_iter=50, normalized=True)
    hubs = sorted(hubs.items(), key=lambda v: (v[1], v[0]), reverse=True)
    print("Hub Scores: ", hubs)
    authorities = sorted(authorities.items(), key=lambda v: (v[1], v[0]), reverse=True)
    print("Authority Scores: ", authorities)



def PageRank():
    G = nx.DiGraph()
    G.add_edges_from(expansion_set)
    pr = nx.pagerank(G, 0.15)
    pagerank_sorted = sorted(pr.items(), key=lambda v: (v[1], v[0]), reverse=True)
    print("Page Rank Score:", pagerank_sorted)


if __name__ == '__main__':
    rootIndex = [27, 68, 123, 175, 327, 620, 761, 763, 905]
    path = "K:/TUK/Sem 2/IRDM/adj.csv"
    HITS(path, rootIndex)
    PageRank()
