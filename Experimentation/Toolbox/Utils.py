import os
import pickle
import random
import re
import networkit as nk
import numpy as np
from sklearn import metrics

pattern = re.compile(r".*lfr_5000/(mk\d+)/(k\d+)/(muw\d+(?:\.\d+)?)/\d+$")
patternRef = re.compile(r".*reference_model(?:_7)?/(mk\d+)(k\d+)(muw\d+(?:\.\d+)?)\.model\d+\.dat$")

def extractParams(path):
    if "reference_model" in path:
        return patternRef.match(path).groups()
    else:
        return pattern.match(path).groups()

def representativeSamples():
    exec_path = os.path.split(os.path.realpath(__file__))[0]
    list_graph = []
    ban_paths =  [os.path.join(exec_path, "../../lfr_5000/mk100/k20/muw0.3/33/"),
                 os.path.join(exec_path, "../../lfr_5000/mk100/k25/muw0.2/45/"),
                 os.path.join(exec_path, "../../lfr_5000/mk100/k25/muw0.3/38/"),
                 os.path.join(exec_path, "../../lfr_5000/mk100/k25/muw0.4/24/"),
                 os.path.join(exec_path, "../../lfr_5000/mk300/k15/muw0.2/20/"),
                 os.path.join(exec_path, "../../lfr_5000/mk300/k15/muw0.3/24/"),
                 os.path.join(exec_path, "../../lfr_5000/mk300/k20/muw0.3/3/"),
                 os.path.join(exec_path, "../../lfr_5000/mk300/k25/muw0.2/37/"),
                 os.path.join(exec_path, "../../lfr_5000/mk500/k25/muw0.2/26/"),
                 os.path.join(exec_path, "../../lfr_5000/mk500/k25/muw0.2/4/")]
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(exec_path, "../../lfr_5000")):
        if "network.dat" in filenames and\
           "community.dat" in filenames and\
           all(not os.path.samefile(dirpath, ban_path) for ban_path in ban_paths):
           list_graph.append(os.path.abspath(dirpath))

    ref = random.sample(set(filter(lambda x: "/muw0.4/" in x, list_graph)), 10) + \
          random.sample(set(filter(lambda x: "/muw0.2/" in x, list_graph)), 10) + \
          random.sample(set(filter(lambda x: "/muw0.3/" in x, list_graph)), 10) + \
          random.sample(set(filter(lambda x: "/k15/" in x, list_graph)), 10) + \
          random.sample(set(filter(lambda x: "/k20/" in x, list_graph)), 10) + \
          random.sample(set(filter(lambda x: "/k25/" in x, list_graph)), 10) + \
          random.sample(set(filter(lambda x: "/mk300/" in x, list_graph)), 10) + \
          random.sample(set(filter(lambda x: "/mk500/" in x, list_graph)), 10) + \
          random.sample(set(filter(lambda x: "/mk100/" in x, list_graph)), 10)
    return ref


def loadings(path, verbose=True):
    if verbose:
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
    if verbose:
        nk.overview(G)

    # loading of communities
    gt_partition = nk.community.readCommunities(os.path.join(path, "community.dat"), format="edgelist-t1")
    # communities properties
    res["numberOfComGroundtruth"] = gt_partition.numberOfSubsets()
    if verbose:
        nk.community.inspectCommunities(gt_partition, G)
        print(f"{gt_partition.numberOfSubsets()} community detected")
    return G, gt_partition, res


def partitionRes(G, gt_partition, detected, method, norm, verbose=True):
    res = {}
    if verbose:
        print(f"__________\nNormalisation :{norm}\nMethod: {method}\n__________")
    res[f"numberOfCom_{method}_{norm}"] = detected.numberOfSubsets()
    NMI = nk.community.NMIDistance().getDissimilarity(G, gt_partition, detected)
    if verbose:
        print(f"{gt_partition.numberOfSubsets()} community detected")
        print(f"NMI:{NMI}")
    res[f"NMI_{method}_{norm}"] = NMI
    ARI = nk.community.AdjustedRandMeasure().getDissimilarity(G, gt_partition, detected)
    res[f"ARI_{method}_{norm}"] = ARI
    if verbose:
        print(f"ARI:{ARI}\n")
        print(f"__________")
    return res


def statNodes(G, gt_partition, edges, addAssort=True, verbose=True):
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
    if verbose:
        print("%d features on %d samples" % tuple(X.shape))
    return X, Y, target, features


def statClassifier(gbm, Y, predictions, verbose=True):
    if verbose:
        print(metrics.classification_report(Y, predictions))
    mat = metrics.confusion_matrix(Y, predictions)
    if verbose:
        print("Confusion matrix:")
        print(mat)
    prec, rec, fmeasure, support = metrics.precision_recall_fscore_support(Y, predictions)
    weights = gbm.feature_importances_
    if verbose:
        print("Importance of features:")
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
