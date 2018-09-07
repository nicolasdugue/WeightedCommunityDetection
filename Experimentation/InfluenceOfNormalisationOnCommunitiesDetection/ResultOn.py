import argparse
import networkit as nk
import os
import pickle
from new_normalization import fake_pmi, glove, iterative_rev_degree_order, pmi, ppmi, standard

norma = {f.__name__: f for f in [glove,
                                 # fake_pmi,
                                 iterative_rev_degree_order,
                                 pmi,
                                 # ppmi,
                                 standard]}
classic_methods = [("Louvain", lambda G: nk.community.detectCommunities(G)),
                   ("Singleton", lambda G: nk.community.ClusteringGenerator().makeSingletonClustering(G)),
                   ("AllInOne", lambda G: nk.community.ClusteringGenerator().makeOneClustering(G)),
                   ("PLP", lambda G: nk.community.detectCommunities(G, nk.community.PLP(G)))]

argparser = argparse.ArgumentParser()
argparser.add_argument("path", help="Directory with network and community", type=str)
args = argparser.parse_args()
path = args.path
print("__LOADINGS__")
# loading of graph

G = nk.graphio.readGraph(os.path.join(path, "network.dat"), weighted=True, fileformat=nk.Format.EdgeListTabOne)
res = dict(numberOfnodes=G.numberOfNodes(), numberOfEdges=G.numberOfEdges())
nk.overview(G)
# loading of communities
evalname = "Groundtruth"
print(f"__{evalname}__")
gt_partition = nk.community.readCommunities(os.path.join(path, "community.dat"), format="edgelist-t1")
nk.community.inspectCommunities(gt_partition, G)
res["numberOfCom" + evalname] = gt_partition.numberOfSubsets()
print(f"{gt_partition.numberOfSubsets()} community detected")

# Classic method
print("__CLASSIC_METHODS__")
for evalname, fdetection in classic_methods:
    print(f"__{evalname}__")
    detected = fdetection(G)
    res["numberOfCom" + evalname] = detected.numberOfSubsets()
    print(f"{gt_partition.numberOfSubsets()} community detected")
    NMI = nk.community.NMIDistance().getDissimilarity(G, gt_partition, detected)
    print(f"NMI:{NMI}")
    res["NMI" + evalname] = NMI
    ARM = nk.community.AdjustedRandMeasure().getDissimilarity(G, gt_partition, detected)
    print(f"ARM:{ARM}")
    res["ARM" + evalname] = ARM

# Normalization
print("__NORMALIZATION__")
for normname, functor in norma.items():
    Gn = functor(G)
    nk.overview(Gn)
    for evalname, fdetection in [("Louvain", nk.community.detectCommunities), ("PLP", lambda Gn: nk.community.detectCommunities(Gn, nk.community.PLP(Gn)))]:
        evalname = normname + "+" + evalname
        print(f"__{evalname}__")
        detected = fdetection(G)
        res["numberOfCom" + evalname] = detected.numberOfSubsets()
        print(f"{gt_partition.numberOfSubsets()} community detected")
        NMI = nk.community.NMIDistance().getDissimilarity(G, gt_partition, detected)
        print(f"NMI:{NMI}")
        res["NMI" + evalname] = NMI
        ARM = nk.community.AdjustedRandMeasure().getDissimilarity(G, gt_partition, detected)
        print(f"ARM:{ARM}")
        res["ARM" + evalname] = ARM

with open(os.path.join(path, "xp1.pickle"), "wb") as file:
    pickle.dump(res, file)
