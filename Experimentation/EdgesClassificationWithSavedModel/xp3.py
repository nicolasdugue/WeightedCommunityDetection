import os
import pickle
import numpy as np

list_graph = []
mk = {}
k = {}
mu = {}
pattern = re.compile(r".*reference_model/mk(\d+)k(\d+)muw(\d+(?:\.\d+)?)\.model(\d+)\.dat$")
nb_graph, gid = 0, 0
for (dirpath, dirnames, filenames) in os.walk("../../lfr_5000"):
    if filenames:
        nb_graph += 1
    if "network.dat" in filenames and "community.dat" in filenames:
        list_graph.append(dirpath)
        dirpath, param = os.path.split(os.path.split(dirpath)[0])
        mu[param] = mu.get(param, []) + [gid]
        dirpath, param = os.path.split(dirpath)
        k[param] = k.get(param, []) + [gid]
        dirpath, param = os.path.split(dirpath)
        mk[param] = mk.get(param, []) + [gid]
        gid += 1
print(f"mising: {len(list_graph)}/{nb_graph}")
# %%
for path in list_graph:
    try:
        with open(os.path.join(path, "xp3.pickle"), "rb") as file:
            reslabel = list(pickle.load(file).keys())
    except FileNotFoundError as e:
        # print(e)
        continue
    break
print(reslabel)
# %%
failled, ldict = 0, []
for path in list_graph:
    try:
        with open(os.path.join(path, "xp3.pickle"), "rb") as file:
            ldict.append(pickle.load(file))
    except FileNotFoundError as e:
        failled += 1
        ldict.append(None)
        # print(e)
        continue
print(f"fail: {failled}/{len(list_graph)}")
for p in reslabel:
    if p == "target" or p == "features":
        print(f"{p}:{ldict[0][p]}")
    else:
        print(f"{p}:{np.mean([d[p] for d in ldict if d], axis=0)}")
# %%
print("__Result__")
for meta_param, meta_param_name in zip([mu], ["mu"]):
    print(f"__Influence of {meta_param_name}__")
    failled, ldict = 0, []
    for valmk, gids in meta_param.items():
        print(f"{meta_param_name}={valmk}")
        for gid in gids:
            try:
                with open(os.path.join(list_graph[gid], "xp3.pickle"), "rb") as file:
                    ldict.append(pickle.load(file))
            except FileNotFoundError as e:
                failled += 1
                # print(e)
                continue
        print(f"fail: {failled}/{len(gids)}")
        for p in ldict[0]:
            if p == "target" or p == "features":
                print(f"{p}:{ldict[0][p]}")
            else:
                print(f"{p}:{np.mean([d[p] for d in ldict if d], axis=0)}")
