import os
import pickle
import random
import re
import argparse
import sys
sys.path.append("../Toolbox")
from Utils import loadings, statNodes
import xgboost as xgb
# %%
argparser = argparse.ArgumentParser()
argparser.add_argument("--addAssort", help="If true assortativity features are used, default=True", action="store_true", default=False)
args = argparser.parse_args()
addAssort = args.addAssort
print(addAssort)
# %%
list_graph = []
nb_graph = 0
for (dirpath, dirnames, filenames) in os.walk("../../lfr_5000"):
    if filenames:
        nb_graph += 1
    if "network.dat" in filenames and\
       "community.dat" in filenames and\
       all(not os.path.samefile(dirpath, banpath) for banpath in ["../../lfr_5000/mk100/k20/muw0.3/33/",
                                                                   "../../lfr_5000/mk100/k25/muw0.2/45/",
                                                                   "../../lfr_5000/mk100/k25/muw0.3/38/",
                                                                   "../../lfr_5000/mk100/k25/muw0.4/24/",
                                                                   "../../lfr_5000/mk300/k15/muw0.2/20/",
                                                                   "../../lfr_5000/mk300/k15/muw0.3/24/",
                                                                   "../../lfr_5000/mk300/k20/muw0.3/3/",
                                                                   "../../lfr_5000/mk300/k25/muw0.2/37/",
                                                                   "../../lfr_5000/mk500/k25/muw0.2/26/",
                                                                   "../../lfr_5000/mk500/k25/muw0.2/4/"]):
        list_graph.append(dirpath)

ref = random.sample(set(filter(lambda x: "/muw0.4/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/muw0.2/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/muw0.3/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/k15/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/k20/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/k25/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/mk300/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/mk500/" in x, list_graph)), 10) + \
      random.sample(set(filter(lambda x: "/mk100/" in x, list_graph)), 10)
print(f"mising: {len(list_graph)}/{nb_graph}")
# %%
pattern = re.compile(r".*lfr_5000/mk(\d+)/k(\d+)/muw(\d+(?:\.\d+)?)/\d+$")
for i, path in enumerate(ref):
    print(path)
    curmk, curk, curmuw = pattern.match(path).groups()
    G, gt_partition, _ = loadings(path)

    edges = G.edges()
    X, Y, target, features = statNodes(G, gt_partition, edges, addAssort=addAssort)
    if addAssort:
        gbm = xgb.XGBClassifier(max_depth=8, n_estimators=300, learning_rate=0.05).fit(X, Y)
        with open(os.path.join("reference_model_7", f"mk{curmk}k{curk}muw{curmuw}.model{i}.dat"), "wb") as file:
            pickle.dump(gbm, file)
    else:
        gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X, Y)
        with open(os.path.join("reference_model", f"mk{curmk}k{curk}muw{curmuw}.model{i}.dat"), "wb") as file:
            pickle.dump(gbm, file)
    # %%
