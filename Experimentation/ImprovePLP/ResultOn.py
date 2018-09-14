from functools import partial
import sys
sys.path.append("../Toolbox")
import argparse
import networkit as nk
from new_normalization import glove
from Utils import loadings, partitionRes, statNodes, statClassifier, extractParams
from PLPvariations import PLP, PLPImproveNico
import os
import xgboost as xgb
import pickle
path = "/home/vconnes/WeightedCommunityDetection/lfr_5000/mk100/k20/muw0.4/4/"
addAssort = True
verbose=False
# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("path", help="Directory with network and community", type=str)
argparser.add_argument("--addAssort", help="If true assortativity features are used, default=True", action="store_true", default=False)
argparser.add_argument("--noVerbose", help="If true assortativity features are used, default=True", action="store_true", default=False)
args = argparser.parse_args()
path = args.path
addAssort = args.addAssort
verbose = not args.noVerbose
# %%
if addAssort:
    refs = [file.path for file in os.scandir("../EdgesClassificationWithSavedModel/reference_model_7")]
else:
    refs = [file.path for file in os.scandir("../EdgesClassificationWithSavedModel/reference_model")]
G, gt_partition, _ = loadings(path, verbose=verbose)
tot = G.totalEdgeWeight()

# %%
norma = dict()
# norma["glovexmax50alpha0.2"] = partial(glove, xmax=50, alpha=float(0.2))
# norma["glovexmax40alpha0.2"] = partial(glove, xmax=40, alpha=float(0.2))
norma["glovexmax30alpha0.3"] = partial(glove, xmax=30, alpha=float(0.3))
detectorLouv = lambda G: nk.community.detectCommunities(G)
detector = lambda G: nk.community.detectCommunities(G, nk.community.PLP(G))
res = {}
detected = detectorLouv(G)
res.update(partitionRes(G, gt_partition, detected, "Louvain", "", verbose=verbose))
detected = detector(G)
res.update(partitionRes(G, gt_partition, detected, "PLP", "", verbose=verbose))
res.update(partitionRes(G, gt_partition, PLP(G), "ownPLP", "", verbose=verbose))
# %%
for name, norm in norma.items():
    print(name, "\n")
    Gn = norm(G)
    detected = detector(Gn)
    res.update(partitionRes(G, gt_partition, detected, "PLP", name, verbose=verbose))
# %%
edges = G.edges()
X, Y, target, features = statNodes(G, gt_partition, edges, addAssort=addAssort, verbose=verbose)
if addAssort:
    gbm = xgb.XGBClassifier(max_depth=8, n_estimators=300, learning_rate=0.05).fit(X, Y)
else:
    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X, Y)
# %%
predictions = gbm.predict(X)
proba_pred = gbm.predict_proba(X)
res.update(target=target, features=features)
res.update({"own_" + k:v for k, v in statClassifier(gbm, Y, predictions, verbose=verbose).items()})
inter, intra = {}, {}
for n in G.nodes():
    inter[n], intra[n] = [], []
cpt = 0
for (u, v) in edges:
    if proba_pred[cpt][0] > 0.7:
        inter[u].append(v)
        inter[v].append(u)
    if proba_pred[cpt][1] > 0.7:
        intra[u].append(v)
        intra[v].append(u)
    cpt += 1
res.update({"own_" + k:v for k, v in partitionRes(G, gt_partition, PLPImproveNico(G, inter=inter, intra=intra), "PLPnico", "", verbose=verbose).items()})
# %%
refres = {}
for i, ref in enumerate(refs):
    params = extractParams(ref) + ("all",)
    with open(ref, "rb") as model:
        gbm = pickle.load(model)
    print(f"[{i+1}/{len(refs)}]: {ref}")
    predictions = gbm.predict(X)
    proba_pred = gbm.predict_proba(X)
    statClass = statClassifier(gbm, Y, predictions, verbose=verbose)
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
    statPart = partitionRes(G, gt_partition, PLPImproveNico(G, inter=inter, intra=intra), "PLPnico", "", verbose=verbose)
    for param in params:
        try:
            refres["train=" + param] += [{**statPart, **statClass}]
        except KeyError:
            refres["train=" + param] = [{**statPart, **statClass}]
# %%
for param, ldict in refres.items():
    for sk in ldict[0].keys():
        res[param + "_" + sk]= [d[sk] for d in ldict]
# %%
if addAssort:
    with open(os.path.join(path, "xp4_7.pickle"), "wb") as file:
        pickle.dump(res, file)
else:
    with open(os.path.join(path, "xp4.pickle"), "wb") as file:
        pickle.dump(res, file)
