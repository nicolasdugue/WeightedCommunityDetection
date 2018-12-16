import sys
sys.path.append("../Toolbox")
import argparse
import os
import pickle
from sklearn import metrics
from Utils import loadings, statNodes, statClassifier, extractParams
path = "/home/vconnes/WeightedCommunityDetection/lfr_5000/mk100/k20/muw0.4/5/"
addAssort=True
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
    refs = [file.path for file in os.scandir("reference_model_7")]
else:
    refs = [file.path for file in os.scandir("reference_model")]
# %%
G, gt_partition, _ = loadings(path, verbose=verbose)
tot = G.totalEdgeWeight()
# %%
edges = G.edges()
X, Y, target, features = statNodes(G, gt_partition, edges,
                                   addAssort=addAssort, verbose=verbose)

# %%
res = dict()
for i, ref in enumerate(refs):
    params = extractParams(ref) + ("all",)
    with open(ref, "rb") as model:
        gbm = pickle.load(model)
    print(f"[{i+1}/{len(refs)}]: {ref}")
    predictions = gbm.predict(X)
    stat = statClassifier(gbm, Y, predictions, verbose=verbose)
    for param in params:
        try:
            res["train=" + param] += [stat]
        except KeyError:
            res["train=" + param] = [stat]
# %%
newres = {}
for param, ldict in res.items():
    for sk in ldict[0].keys():
        newres[param + "_" + sk]= [d[sk] for d in ldict]
newres.update(dict(target=target, features=features))
res = newres
# %%
if addAssort:
    with open(os.path.join(path, "xp3_7.pickle"), "wb") as file:
        pickle.dump(res, file)
else:
    with open(os.path.join(path, "xp3.pickle"), "wb") as file:
        pickle.dump(res, file)
