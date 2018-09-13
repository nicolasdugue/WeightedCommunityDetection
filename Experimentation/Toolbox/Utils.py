import os
import pickle
import networkit as nk
import numpy as np
from sklearn import metrics

def loadings(path):
    print("__LOADINGS__")
    # loading of graph
    G = nk.graphio.readGraph(os.path.join(path, "network.dat"), weighted=True, fileformat=nk.Format.EdgeListTabOne)
    # remove of nul edges and nul degree nodes
    removed = []
    for u, v in G.edges():
        if G.weight(u, v) == 0:
            removed.append((u, v))
    res = dict(numberOfnodes=G.numberOfNodes(), numberOfEdges=G.numberOfEdges(),
               percentOfNulWeight=len([1 for u, v in G.edges() if G.weight(u, v) == 0])/G.numberOfEdges())
    for (u, v) in removed:
        G.removeEdge(u, v)
    # graph properties
    nk.overview(G)

    # loading of communities
    gt_partition = nk.community.readCommunities(os.path.join(path, "community.dat"), format="edgelist-t1")
    # communities properties
    nk.community.inspectCommunities(gt_partition, G)
    res["numberOfComGroundtruth"] = gt_partition.numberOfSubsets()
    print(f"{gt_partition.numberOfSubsets()} community detected")
    return G, gt_partition, res


def partitionRes(G, gt_partition, detected, method, norm):
    res = {}
    print(f"__________\nNormalisation :{norm}\nMethod: {method}\n__________")
    res[f"numberOfCom_{method}_{norm}"] = detected.numberOfSubsets()
    NMI = nk.community.NMIDistance().getDissimilarity(G, gt_partition, detected)
    print(f"{gt_partition.numberOfSubsets()} community detected")
    print(f"NMI:{NMI}")
    res[f"NMI_{method}_{norm}"] = NMI
    ARI = nk.community.AdjustedRandMeasure().getDissimilarity(G, gt_partition, detected)
    res[f"ARI_{method}_{norm}"] = ARI
    print(f"ARI:{ARI}\n")
    print(f"__________")
    return res


def statNodes(G, gt_partition, edges, addAssort=True):
    deg_min = []
    deg_max = []
    clust_min = []
    clust_max = []
    weight = []
    inside = []
    if addAssort:
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
        if addAssort:
            minnode, maxnode = sorted([u, v], key=G.weightedDegree)
            deg_moyn_min.append(np.mean([G.weightedDegree(n) for n in G.neighbors(minnode)]))
            deg_moyn_max.append(np.mean([G.weightedDegree(n) for n in G.neighbors(maxnode)]))

        if gt_partition.subsetOf(u) == gt_partition.subsetOf(v):
            inside.append(1)
        else:
            inside.append(0)
    X = np.array([deg_min, deg_max, clust_min, clust_max, weight, deg_moyn_min, deg_moyn_max]) if addAssort else np.array([deg_min, deg_max, clust_min, clust_max, weight])
    Y = inside
    X = X.transpose()
    target = ["outside", "inside"]
    features = ["deg_min", "deg_max", "clust_min", "clust_max", "weight"]
    if addAssort:
        features += ["deg_moyn_min", "deg_moyn_max" ]
    # print(X.shape)
    print("%d features on %d samples" % tuple(X.shape))
    return X, Y, target, features


def statClassifier(gbm, Y, predictions):
    print(metrics.classification_report(Y, predictions))
    mat = metrics.confusion_matrix(Y, predictions)
    print("Confusion matrix:")
    print(mat)
    prec, rec, fmeasure, support = metrics.precision_recall_fscore_support(Y, predictions)
    print("Importance of features:")
    weights = gbm.feature_importances_
    print(weights)
    res = dict(precision=prec, recall=rec, f1=fmeasure, support=support, confmat=mat, weights=weights)
    return res


def architecture(picklename):
    list_graph = []
    mk = {}
    k = {}
    mu = {}

    nb_graph, gid = 0, 0
    for (dirpath, dirnames, filenames) in os.walk("../../lfr_5000"):
        if filenames:
            nb_graph += 1
        if "network.dat" in filenames and "community.dat" in filenames:
            list_graph.append(dirpath)
            dirpath, param = os.path.split(os.path.split(dirpath)[0])
            mu[param] = mu.get(param, []) + [gid]
            dirpath, param = os.path.split(dirpath)
            k[param] = k.get(param, []) + [gid]
            dirpath, param = os.path.split(dirpath)
            mk[param] = mk.get(param, []) + [gid]
            gid += 1
    print(f"mising: {nb_graph-len(list_graph)}/{nb_graph}")
    for path in list_graph:
        try:
            with open(os.path.join(path, picklename), "rb") as file:
                reslabel = list(pickle.load(file).keys())
        except FileNotFoundError as e:
            # print(e)
            continue
        break
    failled, ldict = 0, []
    for path in list_graph:
        try:
            with open(os.path.join(path, picklename), "rb") as file:
                ldict.append(pickle.load(file))
        except FileNotFoundError as e:
            failled += 1
            ldict.append(None)
            # print(e)
            continue
    print(f"fail: {failled}/{len(list_graph)}")
    return ldict, list_graph, mk, k, mu, reslabel
