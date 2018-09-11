from functools import partial
import argparse
import networkit as nk
import os
import pickle
from new_normalization import fake_pmi, glove, iterative_rev_degree_order, pmi, ppmi, standard

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


def BCC(G):
    name = G.getName()
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    iccmax = max(cc.getPartition().subsetSizeMap().items(), key=lambda x: x[1])[0]
    Gc = G.subgraphFromNodes(cc.getPartition().getMembers(iccmax))
    gmap = nk.graph.GraphTools.getContinuousNodeIds(Gc)
    Gc = nk.graph.GraphTools.getCompactedGraph(Gc, gmap)
    Gc.setName(name + "cc")
    return Gc


# %%
argparser = argparse.ArgumentParser()
argparser.add_argument("path", help="Directory with network and community", type=str)
args = argparser.parse_args()
path = args.path
print("__LOADINGS__")
# %%
# loading of graph
# path = "/home/vconnes/WeightedCommunityDetection/lfr_5000/mk100/k20/muw0.4/4/"
loadG = nk.graphio.readGraph(os.path.join(path, "network.dat"), weighted=True, fileformat=nk.Format.EdgeListTabOne)
removed = []
for u, v in loadG.edges():
    if loadG.weight(u, v) == 0:
        removed.append((u, v))
res = dict(numberOfnodes=loadG.numberOfNodes(), numberOfEdges=loadG.numberOfEdges(),
           percentOfNulWeight=len([1 for u, v in loadG.edges() if loadG.weight(u, v) == 0])/loadG.numberOfEdges())
for (u, v) in removed:
    loadG.removeEdge(u, v)
nk.overview(loadG)
tot = loadG.totalEdgeWeight()
print(tot)
# loading of communities
evalname = "Groundtruth"
print(f"__{evalname}__")
gt_partition = nk.community.readCommunities(os.path.join(path, "community.dat"), format="edgelist-t1")
nk.community.inspectCommunities(gt_partition, loadG)
res["numberOfCom" + evalname] = gt_partition.numberOfSubsets()
print(f"{gt_partition.numberOfSubsets()} community detected")
# %%
# Classic method
print("__CLASSIC_METHODS__")
for evalname, fdetection in classic_methods:
    print(f"__{evalname}__")
    detected = fdetection(loadG)
    res["numberOfCom" + evalname] = detected.numberOfSubsets()
    print(f"{gt_partition.numberOfSubsets()} community detected")
    NMI = nk.community.NMIDistance().getDissimilarity(loadG, gt_partition, detected)
    print(f"NMI:{NMI}")
    res["NMI" + evalname] = NMI
    ARM = nk.community.AdjustedRandMeasure().getDissimilarity(loadG, gt_partition, detected)
    print(f"ARM:{ARM}")
    res["ARM" + evalname] = ARM
# %%
# Normalization
print("__NORMALIZATION__")
for normname, functor in norma.items():
    print(f"__{normname}__")
    Gn = BCC(functor(loadG))
    nk.overview(Gn)
    print("tot: ", Gn.totalEdgeWeight())
    assert tot == loadG.totalEdgeWeight()
    for evalname, fdetection in [("Louvain", nk.community.detectCommunities), ("PLP", lambda G: nk.community.detectCommunities(G, nk.community.PLP(G)))]:
        evalname = normname + "+" + evalname
        print(f"__{evalname}__")
        if Gn.totalEdgeWeight() != 0:
            detected = fdetection(Gn)
            res["numberOfCom" + evalname] = detected.numberOfSubsets()
            NMI = nk.community.NMIDistance().getDissimilarity(Gn, gt_partition, detected)
            print(f"{gt_partition.numberOfSubsets()} community detected")
            print(f"NMI:{NMI}")
            res["NMI" + evalname] = NMI
            ARM = nk.community.AdjustedRandMeasure().getDissimilarity(Gn, gt_partition, detected)
            print(f"ARM:{ARM}")
            res["ARM" + evalname] = ARM
        else:
            ARI, NMI = 1, 1
            print(f"1 community detected due to total edge weight equal 0")
            print(f"NMI:{NMI}")
            print(f"ARM:{ARM}")
            res["numberOfCom" + evalname] = 1
            res["NMI" + evalname] = NMI
            res["ARM" + evalname] = ARM

with open(os.path.join(path, "xp1.pickle"), "wb") as file:
    pickle.dump(res, file)
