import sys
sys.path.append("../Toolbox")
import argparse
import os
import pickle
from sklearn import metrics
import networkit as nk
import numpy as np
from Utils import loadings, statNodes, statClassifier
path = "/home/vconnes/WeightedCommunityDetection/lfr_5000/mk100/k20/muw0.4/5/"
addAssort=True
# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("path", help="Directory with network and community", type=str)
argparser.add_argument("--addAssort", help="If true assortativity features are used, default=True", action="store_true", default=False)
args = argparser.parse_args()
path = args.path
addAssort = args.addAssort
# %%
if addAssort:
    refs = [file.path for file in os.scandir("reference_model_7")]
else:
    refs = [file.path for file in os.scandir("reference_model")]
# %%
G, gt_partition, _ = loadings(path)
tot = G.totalEdgeWeight()
# %%
edges = G.edges()
X, Y, target, features = statNodes(G, gt_partition, edges, addAssort=addAssort)

# %%
refres = dict()
for i, ref in enumerate(refs):
    with open(ref, "rb") as model:
        gbm = pickle.load(model)
    print(f"[{i+1}/{len(refs)}]: {ref}")
    predictions = gbm.predict(X)
    refres[ref] = statClassifier(gbm, Y, predictions).update(target=target, features=features)
# %%
if addAssort:
    with open(os.path.join(path, "xp3_7.pickle"), "wb") as file:
        pickle.dump(refres, file)
else:
    with open(os.path.join(path, "xp3.pickle"), "wb") as file:
        pickle.dump(refres, file)
