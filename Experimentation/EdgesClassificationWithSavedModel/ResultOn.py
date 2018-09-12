import argparse
import os
import pickle
from sklearn import metrics
import networkit as nk
import numpy as np
# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("path", help="Directory with network and community", type=str)
args = argparser.parse_args()
path = args.path
# %%
refs = [file.path for file in os.scandir("reference_model")]
# %%
# path = "/home/vconnes/WeightedCommunityDetection/lfr_5000/mk100/k20/muw0.4/5/"

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
# loading of model
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

target = ["outside", "inside"]
features = ["deg_min", "deg_max", "clust_min", "clust_max", "weight"]
features += ["deg_moyn_min", "deg_moyn_max" ]
X = np.array([deg_min, deg_max, clust_min, clust_max, weight])
Y = inside
X = X.transpose()
samples, features = X.shape

# %%
refres = {}
for i, ref in enumerate(refs):
    with open(ref, "rb") as model:
        gbm = pickle.load(model)
    print(f"[{i+1}/{len(refs)}]: {ref}")
    predictions = gbm.predict(X)
    print(metrics.classification_report(Y, predictions))
    mat = metrics.confusion_matrix(Y, predictions)
    print("Confusion matrix:")
    print(mat)
    prec, rec, fmeasure, support = metrics.precision_recall_fscore_support(Y, predictions)
    print("Importance of features:")
    weights = gbm.feature_importances_
    print(weights)
    res = dict(target=target, precision=prec, recall=rec, f1=fmeasure, support=support, confmat=mat,
               features=features, weights=weights)

    refres[ref] = res
# %%
with open(os.path.join(path, "xp3_7.pickle"), "wb") as file:
    pickle.dump(refres, file)
