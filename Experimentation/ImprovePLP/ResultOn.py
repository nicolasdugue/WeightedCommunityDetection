from functools import partial
import sys
sys.path.append("../Toolbox")
import argparse
from sklearn import metrics
import networkit as nk
from new_normalization import glove
from Utils import loadings, partitionRes, statNodes, statClassifier
from PLPvariations import PLP, PLPImproveNico
import os
import xgboost as xgb
import pickle
path = "/home/vconnes/WeightedCommunityDetection/lfr_5000/mk100/k20/muw0.4/4/"
addAssort = True
# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("path", help="Directory with network and community", type=str)
argparser.add_argument("--addAssort", help="If true assortativity features are used, default=True", action="store_true", default=False)
args = argparser.parse_args()
path = args.path
addAssort = args.addAssort
# %%
if addAssort:
    refs = [file.path for file in os.scandir("../EdgesClassificationWithSavedModel/reference_model_7")]
else:
    refs = [file.path for file in os.scandir("../EdgesClassificationWithSavedModel/reference_model")]
G, gt_partition, _ = loadings(path)
tot = G.totalEdgeWeight()

# %%
norma = dict()
# norma["glovexmax50alpha0.2"] = partial(glove, xmax=50, alpha=float(0.2))
# norma["glovexmax40alpha0.2"] = partial(glove, xmax=40, alpha=float(0.2))
norma["glovexmax30alpha0.3"] = partial(glove, xmax=30, alpha=float(0.3))
detectorLouv = lambda G: nk.community.detectCommunities(G)
detector = lambda G: nk.community.detectCommunities(G, nk.community.PLP(G))
refres = {}
detected = detectorLouv(G)
refres.update(partitionRes(G, gt_partition, detected, "Louvain", ""))
detected = detector(G)
refres.update(partitionRes(G, gt_partition, detected, "PLP", ""))
refres.update(partitionRes(G, gt_partition, PLP(G), "ownPLP", ""))
# %%
for name, norm in norma.items():
    print(name, "\n")
    Gn = norm(G)
    detected = detector(Gn)
    refres.update(partitionRes(G, gt_partition, detected, "PLP", name))
# %%
edges = G.edges()
X, Y, target, features = statNodes(G, gt_partition, edges, addAssort=addAssort)
if addAssort:
    gbm = xgb.XGBClassifier(max_depth=8, n_estimators=300, learning_rate=0.05).fit(X, Y)
else:
    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X, Y)

# predictions = gbm.predict(X)
proba_pred = gbm.predict_proba(X)
res = {}
# res.update(statClassifier(gbm, Y, predictions))
# res.update(target=target, features=features)
# print(metrics.classification_report(Y, predictions))
inter = dict()
intra = dict()
for node in G.nodes():
    inter[node] = []
    intra[node] = []
cpt = 0
for (u, v) in edges:
    if proba_pred[cpt][0] > 0.7:
        inter[u].append(v)
        inter[v].append(u)
    if proba_pred[cpt][1] > 0.7:
        intra[u].append(v)
        intra[v].append(u)
    cpt += 1
# %%
res.update(partitionRes(G, gt_partition, PLPImproveNico(G, inter=inter, intra=intra), "PLPnico", ""))
refres["own"] = res
# %%
for i, ref in enumerate(refs):
    res = {}
    with open(ref, "rb") as model:
        gbm = pickle.load(model)
    print(f"[{i+1}/{len(refs)}]: {ref}")
    predictions = gbm.predict(X)
    proba_pred = gbm.predict_proba(X)
    res.update(statClassifier(gbm, Y, predictions))
    res.update(target=target, features=features)
    print(metrics.classification_report(Y, predictions))
    inter = dict()
    intra = dict()
    for node in G.nodes():
        inter[node] = []
        intra[node] = []
    cpt = 0
    for (u, v) in edges:
        if proba_pred[cpt][0] > 0.7:
            inter[u].append(v)
            inter[v].append(u)
        if proba_pred[cpt][1] > 0.7:
            intra[u].append(v)
            intra[v].append(u)
        cpt += 1
    res.update(partitionRes(G, gt_partition, PLPImproveNico(G, inter=inter, intra=intra), "PLPnico", ""))
    refres[ref]=res
# %%
if addAssort:
    with open(os.path.join(path, "xp4_7.pickle"), "wb") as file:
        pickle.dump(refres, file)
else:
    with open(os.path.join(path, "xp4.pickle"), "wb") as file:
        pickle.dump(refres, file)
