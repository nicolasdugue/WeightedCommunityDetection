import argparse
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import networkit as nk
import numpy as np
import xgboost as xgb
# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("path", help="Directory with network and community", type=str)
args = argparser.parse_args()
path = args.path
# %%
# path = "/home/vconnes/WeightedCommunityDetection/lfr_5000/mk100/k20/muw0.4/4/"
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

    if gt_partition.subsetOf(u) == gt_partition.subsetOf(v):
        inside.append(1)
    else:
        inside.append(0)

target = ["outside", "inside"]
features = ["deg_min", "deg_max", "clust_min", "clust_max", "weight"]
X = np.array([deg_min, deg_max, clust_min, clust_max, weight])
Y = inside
X = X.transpose()
samples, features = X.shape
print(f"{features} features on {samples} samples")

print(f"Trainning set:{len(X_train)} samples")
print(f"Testing set:{len(X_test)} samples")

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X, Y)
predictions = gbm.predict_proba(X_test)
