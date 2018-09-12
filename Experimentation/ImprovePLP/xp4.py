from functools import partial
import sys
sys.path.append("../InfluenceOfNormalisationOnCommunitiesDetection")
from  new_normalization import glove
import argparse
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import networkit as nk
import numpy as np
import xgboost as xgb
import random
import collections
# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("path", help="Directory with network and community", type=str)
args = argparser.parse_args()
path = args.path
# %%
path = "/home/vconnes/WeightedCommunityDetection/lfr_5000/mk100/k20/muw0.4/4/"
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
res = {}
gt_partition = nk.community.readCommunities(os.path.join(path, "community.dat"), format="edgelist-t1")
nk.community.inspectCommunities(gt_partition, G)
res["numberOfComGroundtruth"] = gt_partition.numberOfSubsets()
print(f"{gt_partition.numberOfSubsets()} community detected")

# %%
norma = dict()
# norma["glovexmax50alpha0.2"] = partial(glove, xmax=50, alpha=float(0.2))
# norma["glovexmax40alpha0.2"] = partial(glove, xmax=40, alpha=float(0.2))
norma["glovexmax30alpha0.3"] = partial(glove, xmax=30, alpha=float(0.3))
detector = lambda G: nk.community.detectCommunities(G, nk.community.PLP(G))

# %%
detected = detector(G)
res["numberOfComPLP"] = detected.numberOfSubsets()
NMI = nk.community.NMIDistance().getDissimilarity(G, gt_partition, detected)
print(f"{gt_partition.numberOfSubsets()} community detected")
print(f"NMI:{NMI}")
res["NMIPLP"] = NMI
ARI = nk.community.AdjustedRandMeasure().getDissimilarity(G, gt_partition, detected)
res["ARIPLP"] = ARI
print(f"ARI:{ARI}")
# %%
for name, norm in norma.items():
    print(name, "\n")
    Gn = norm(G)
    # nk.overview(Gn)
    detected = detector(Gn)
    res["numberOfComPLP"] = detected.numberOfSubsets()
    NMI = nk.community.NMIDistance().getDissimilarity(G, gt_partition, detected)
    print(f"{gt_partition.numberOfSubsets()} community detected")
    print(f"NMI:{NMI}")
    res["NMIPLP"] = NMI
    ARI = nk.community.AdjustedRandMeasure().getDissimilarity(G, gt_partition, detected)
    res["ARIPLP"] = ARI
    print(f"ARI:{ARI}\n")

# %%
edges = G.edges()
deg_min = []
deg_max = []
clust_min = []
clust_max = []
weight = []
inside = []
deg_moyn_min, deg_moyn_max = [], []
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
X = np.array([deg_min, deg_max, clust_min, clust_max, weight, deg_moyn_min, deg_moyn_max])
Y = inside
X = X.transpose()
samples, features = X.shape
print(f"{features} features on {samples} samples")
# %%
gbm = xgb.XGBClassifier(max_depth=8, n_estimators=300, learning_rate=0.05).fit(X, Y)
probapred = gbm.predict_proba(X)
predictions = gbm.predict(X)
print(metrics.classification_report(Y, predictions))
# %%
inter = dict()
intra = dict()
for node in G.nodes():
    inter[node] = []
    intra[node] = []
cpt = 0
for (u, v) in edges:
    if predictions[cpt][1] > 0.7:
        inter[u].append(v)
        inter[v].append(u)
    if predictions[cpt][0] > 0.7:
        intra[u].append(v)
        intra[v].append(u)
    cpt += 1

# %%
def PLP(G):
    dicoLabels, nbIter, nodes = dict(), 0,  G.nodes()
    # We start with the partiton induced by the classifier
    dicoLabels = {n:i for i, n in enumerate(nodes)}

    change = True
    while change:
        ordre = list(range(len(nodes)))
        random.shuffle(ordre)
        for n in ordre:
            voisins = set(G.neighbors(n))
            labels = [dicoLabels[v] for v in voisins]
            c = collections.Counter(labels)
            old_label_count = c[dicoLabels[n]]
            newlabel, newcount = c.most_common(1)[0]
            nb_change = 0
            if newlabel != dicoLabels[n] and newcount > old_label_count:
                dicoLabels[n] = newlabel
                nb_change += 1
            print("Nb de changements", nb_change)
            if nb_change == 0 or nbIter > 100:
                change = False
    continousCid = {c, i for i, c in enumerate(sorted(list(set(dicoLabels.values()))))}
    return nk.community.Partition(len(nodes),[[continousCid[dicoLabels[n]] for n in sorted(dicoLabels)]])

def PLP(G, inter, intra):
    dicoLabels = dict()
    nbIter = 0

    nodes = G.nodes()
    cpt = 0
    for n in nodes:
        if n not in dicoLabels:
            dicoLabels[n] = cpt
            voisins = intra[n]
            for v in voisins:
                dicoLabels[n] = cpt
            cpt += 1
    change = True
    while change:
        dicoNew = dict()
        ordre = list(range(len(nodes)))
        random.shuffle(ordre)
        for n in ordre:
            voisins = set(G.neighbors(n))
            not_voisins= set(inter[n])
            voisins = voisins.difference(not_voisins)
            if (len(voisins) > 0):
                labels = [dicoLabels[v] for v in voisins]
                c = collections.Counter(labels)
                old_label_count = c[dicoLabels[n]]
                label = c.most_common(1)[0][0]
                if label != dicoLabels[n] and c.most_common(1)[0][1] > old_label_count:
                    dicoNew[n] = label
                    #voisins=intra[n]
                    #for v in voisins:
                    #    dicoNew[v]=label
        for d in dicoNew:
            dicoLabels[d] = dicoNew[d]
        changement = len(dicoNew)
        nbIter += 1
        print("Nb de changements", changement)
        if (nbIter > 100):
            change = False
    return dicoLabels
