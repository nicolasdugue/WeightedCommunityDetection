import networkit as nk
import os
import np

list_graph = []
mk = {}
k = {}
mu = {}

gid = 0
for (dirpath, dirnames, filenames) in os.walk("../../lfr_5000"):
    if filenames:
        list_graph.append(dirpath)
        dirpath, param = os.path.split(os.path.split(dirpath)[0])
        mu[param] = mu.get(param, []) + [gid]
        dirpath, param = os.path.split(dirpath)
        k[param] = k.get(param, []) + [gid]
        dirpath, param = os.path.split(dirpath)
        mk[param] = mk.get(param, []) + [gid]
        gid += 1

ldict = []
print("__Result__")
for meta_param in [mk, mu, k]:
    print(f"__Influence of {meta_param.__name__}__")
    for valmk, gids in meta_param.items():
        print(f"{meta_param.__name__}={valmk}")
        for gid in gids:
            with open(os.path.join(list_graph[gid], "xp2.pickle"), "w") as file:
                ldict.append(file.load(file))
        for p in ldict[0]:
            print(f"{p}:{np.mean([d[p] for d in ldict if d])}")
