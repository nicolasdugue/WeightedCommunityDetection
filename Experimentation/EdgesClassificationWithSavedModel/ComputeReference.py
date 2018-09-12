import os
import pickle
import random
import re
from sklearn.model_selection import train_test_split
from sklearn import metrics
import networkit as nk
import numpy as np
import xgboost as xgb
# %%

list_graph = []

nb_graph = 0
for (dirpath, dirnames, filenames) in os.walk("../../lfr_5000"):
    if filenames:
        nb_graph += 1
    if "network.dat" in filenames and\
       "community.dat" in filenames and\
       all(not os.path.samefile(dirpath, banpath) for banpath in ["../../lfr_5000/mk100/k20/muw0.3/33/",
                                                                   "../../lfr_5000/mk100/k25/muw0.2/45/",
                                                                   "../../lfr_5000/mk100/k25/muw0.3/38/",
                                                                   "../../lfr_5000/mk100/k25/muw0.4/24/",
                                                                   "../../lfr_5000/mk300/k15/muw0.2/20/",
                                                                   "../../lfr_5000/mk300/k15/muw0.3/24/",
                                                                   "../../lfr_5000/mk300/k20/muw0.3/3/",
                                                                   "../../lfr_5000/mk300/k25/muw0.2/37/",
                                                                   "../../lfr_5000/mk500/k25/muw0.2/26/",
                                                                   "../../lfr_5000/mk500/k25/muw0.2/4/"]):
        list_graph.append(dirpath)

ref = random.sample(set(filter(lambda x: "/muw0.4/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/muw0.2/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/muw0.3/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/k15/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/k20/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/k25/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/mk300/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/mk500/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/mk100/" in x, list_graph)), 10)
print(f"mising: {len(list_graph)}/{nb_graph}")
# %%
pattern = re.compile(r".*lfr_5000/mk(\d+)/k(\d+)/muw(\d+(?:\.\d+)?)/\d+$")
for i, path in enumerate(ref):
    print(path)
    curmk, curk, curmuw = pattern.match(path).groups()
    print("__LOADINGS__")
    # loading of graph
    G = nk.graphio.readGraph(os.path.join(path, "network.dat"), weighted=True, fileformat=nk.Format.EdgeListTabOne)
    removed = []
    for u, v in G.edges():
        if G.weight(u, v) == 0:
            removed.append((u, v))
    res = dict(numberOfnodes=G.numberOfNodes(), numberOfEdges=G.numberOfEdges(),
               percentOfNulWeight=len([1 for u, v in G.edges() if G.weight(u, v) == 0])/G.numberOfEdges())
    for (u, v) in removed:
        G.removeEdge(u, v)
    nk.overview(G)
    tot = G.totalEdgeWeight()
    print(tot)
    # loading of communities
    evalname = "Groundtruth"
    print(f"__{evalname}__")
    gt_partition = nk.community.readCommunities(os.path.join(path, "community.dat"), format="edgelist-t1")
    nk.community.inspectCommunities(gt_partition, G)
    res["numberOfCom" + evalname] = gt_partition.numberOfSubsets()
    print(f"{gt_partition.numberOfSubsets()} community detected")

    edges = G.edges()
    deg_min = []
    deg_max = []
    clust_min = []
    clust_max = []
    deg_moyn_min, deg_moyn_max = [], []
    weight = []
    inside = []
    cc = nk.centrality.LocalClusteringCoefficient(G).run().scores()

    for (u, v) in edges:
        degU, degV = G.weightedDegree(u), G.weightedDegree(v)
        clustU, clustV = cc[u], cc[v]

        deg_min.append(min(degU, degV))
        deg_max.append(max(degU, degV))
        clust_min.append(min(clustU, clustV))
        clust_max.append(max(clustU, clustV))
        weight.append(G.weight(u, v))
        minnode, maxnode = sorted([u, v], key=G.weightedDegree)
        deg_moyn_min.append(np.mean([G.weightedDegree(n) for n in G.neighbors(minnode)]))
        deg_moyn_max.append(np.mean([G.weightedDegree(n) for n in G.neighbors(maxnode)]))


        if gt_partition.subsetOf(u) == gt_partition.subsetOf(v):
            inside.append(1)
        else:
            inside.append(0)

    X = np.array([deg_min, deg_max, clust_min, clust_max, weight
                  , deg_moyn_min, deg_moyn_max
                  ])
    Y = inside
    X = X.transpose()
    samples, features = X.shape
    print(f"{features} features on {samples} samples")

    gbm = xgb.XGBClassifier(max_depth=7, n_estimators=300, learning_rate=0.05).fit(X, Y)
    with open(os.path.join("reference_model_7", f"mk{curmk}k{curk}muw{curmuw}.model{i}.dat"), "wb") as file:
        pickle.dump(gbm, file)
    # %%
