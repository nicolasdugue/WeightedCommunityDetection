from functools import partial
import argparse
import networkit as nk
import os
import pickle
import sys
sys.path.append("../Toolbox")
from new_normalization import fake_pmi, glove, iterative_rev_degree_order, pmi, ppmi, standard
from Utils import loadings, partitionRes
path = "/home/vconnes/WeightedCommunityDetection/lfr_5000/mk100/k20/muw0.4/4/"
# %%
norma = {f.__name__: f for f in [glove,
                                 fake_pmi,
                                 iterative_rev_degree_order,
                                 pmi,
                                 standard,
                                 ppmi,
                                 ]}
norma["glovexmax50alpha0.2"] = partial(glove, xmax=50, alpha=float(0.2))
norma["glovexmax40alpha0.2"] = partial(glove, xmax=40, alpha=float(0.2))
norma["glovexmax30alpha0.3"] = partial(glove, xmax=30, alpha=float(0.3))
classic_methods = [("Louvain", lambda G: nk.community.detectCommunities(G)),
                   ("Singleton", lambda G: nk.community.ClusteringGenerator().makeSingletonClustering(G)),
                   ("AllInOne", lambda G: nk.community.ClusteringGenerator().makeOneClustering(G)),
                   ("PLP", lambda G: nk.community.detectCommunities(G, nk.community.PLP(G)))]
# %%
argparser = argparse.ArgumentParser()
argparser.add_argument("path", help="Directory with network and community", type=str)
args = argparser.parse_args()
path = args.path
# %%
G, gt_partition, res = loadings(path)
tot = G.totalEdgeWeight()
# %%
# Classic method
print("__CLASSIC_METHODS__")
for evalname, fdetection in classic_methods:
    print(f"__{evalname}__")
    detected = fdetection(G)
    res.update(partitionRes(G, gt_partition, detected, evalname, ""))

# %%
# Normalization
print("__NORMALIZATION__")
for normname, functor in norma.items():
    Gn = functor(G)
    nk.overview(Gn)
    print("tot: ", Gn.totalEdgeWeight())
    assert tot == G.totalEdgeWeight()
    for evalname, fdetection in [("Louvain", nk.community.detectCommunities), ("PLP", lambda G: nk.community.detectCommunities(G, nk.community.PLP(G)))]:
        if Gn.totalEdgeWeight() != 0:
            detected = fdetection(Gn)
            res.update(partitionRes(G, gt_partition, detected, evalname, normname))
        else:
            ARI, NMI = 1, 1
            print(f"1 community detected due to total edge weight equal 0")
            print(f"NMI:{NMI}")
            print(f"ARI:{ARI}")
            res[f"numberOfCom_{evalname}_{normname}"] = 1
            res[f"NMI_{evalname}_{normname}"] = NMI
            res[f"ARI_{evalname}_{normname}"] = ARI
print("NMI classement:")
print(sorted([(k, v) for k, v in res.items() if "NMI" in k], key=lambda x: x[1]))
print("ARI classement:")
print(sorted([(k, v) for k, v in res.items() if "ARI" in k], key=lambda x: x[1]))
with open(os.path.join(path, "xp1.pickle"), "wb") as file:
    pickle.dump(res, file)
