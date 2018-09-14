import os
import pickle
import random
import re
import argparse
import sys
sys.path.append("../Toolbox")
from Utils import loadings, statNodes, representativeSamples, extractParams
import xgboost as xgb
# %%
ref = representativeSamples()
for i, path in enumerate(ref):
    curmk, curk, curmuw = extractParams(path)
    G, gt_partition, _ = loadings(path)

    edges = G.edges()
    X, Y, target, features = statNodes(G, gt_partition, edges, addAssort=True)
    gbm = xgb.XGBClassifier(max_depth=8, n_estimators=300, learning_rate=0.05).fit(X, Y)
    with open(os.path.join("reference_model_7", f"{curmk}{curk}{curmuw}.model{i}.dat"), "wb") as file:
        pickle.dump(gbm, file)
    X, Y, target, features = statNodes(G, gt_partition, edges, addAssort=False)
    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X, Y)
    with open(os.path.join("reference_model", f"{curmk}{curk}{curmuw}.model{i}.dat"), "wb") as file:
        pickle.dump(gbm, file)
    if i % 10 == 0 or i == len(ref) - 1 :
        print(f"[{i+1}/{len(ref)}]progress {(i+1)/len(ref)*100:.2f}%")
    # %%
