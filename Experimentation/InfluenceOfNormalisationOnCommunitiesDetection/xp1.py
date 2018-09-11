import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
# %%
list_graph = []
mk = {}
k = {}
mu = {}

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
ldict = []
for path in list_graph:
    try:
        with open(os.path.join(path, "xp1.pickle"), "rb") as file:
            reslabel = list(pickle.load(file).keys())
    except FileNotFoundError as e:
        # print(e)
        continue
    break
print("__Result__")
# %%
colindex = ["mk", "mu", "k"] + reslabel
matres = np.empty([len(list_graph), len(colindex)], dtype=np.float64)
# matres[0] = colindex

failled = 0
for gid, path in enumerate(list_graph):
    # print(path)
    match = re.match(r".*lfr_5000/mk(\d+)/k(\d+)/muw(\d+(?:\.\d+)?)/\d+", path)
    # print(match.groups())
    matres[gid][0:3] = match.groups()
    i = 3
    try:
        with open(os.path.join(list_graph[gid], "xp1.pickle"), "rb") as file:
            resg = pickle.load(file)
    except FileNotFoundError as e:
        # print(gid, e)
        failled += 1
        matres[gid][3:] = [None] * (len(colindex) - 3)
        continue
    for p in reslabel:
        matres[gid][i] = resg[p]
        i += 1
print(f"fail: {failled}/{len(list_graph)}")

# %%
f = plt.figure(figsize=(20, 10))
X = list(mu.keys())
for curve in filter(lambda x: ("pmi" in x or ("+" not in x and "All" not in x and "Sing" not in x)) and ("NMI" in x or "ARM" in x), reslabel):
    Y = []
    for valmk in X:
        ldict = []
        for gid in mu[valmk]:
            try:
                with open(os.path.join(list_graph[gid], "xp1.pickle"), "rb") as file:
                    ldict.append(pickle.load(file))
            except FileNotFoundError as e:
                # print(e)
                continue
        Y.append(np.mean([d[curve] for d in ldict if d]))
    if "NMILouvain" in curve:
        plt.plot(X, Y, label=curve, marker="x", linewidth=5)
    elif "ARMLouvain" in curve:
        plt.plot(X, Y, label=curve, marker="o", linewidth=5)
    elif "NMIPLP" in curve:
        plt.plot(X, Y, label=curve, marker="x", linewidth=5)
    elif "ARMPLP" in curve:
        plt.plot(X, Y, label=curve, marker="o", linewidth=5)
    elif "NMI" in curve:
        plt.plot(X, Y, label=curve, marker="x")
    elif "ARM" in curve:
        plt.plot(X, Y, label=curve, marker="o")
plt.legend()
# plt.ylim(ymin=0, ymax=1)
plt.xlabel("mu")
plt.ylabel("NMI, ARM")

plt.show()

# %%
