import os
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
for path in list_graph:
    try:
        with open(os.path.join(path, "xp2_7.pickle"), "rb") as file:
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
        with open(os.path.join(path, "xp2_7.pickle"), "rb") as file:
            ldict.append(pickle.load(file))
    except FileNotFoundError as e:
        failled += 1
        ldict.append(None)
        # print(e)
        continue
print(f"fail: {failled}/{len(list_graph)}")
# %%

def plotresult(ldict):
    Y = [[], []]
    Yerr = [[], []]
    target = ldict[0]["target"]
    Xlabel = []
    for p in reslabel:
        if p == "target" or p == "features":
            print(f"{p}:{ldict[0][p]}")
        else:
            X = [d[p] for d in ldict if d]
            Xmean = np.mean(X, axis=0)
            Xstd = np.std(X, axis=0)
            print(f"{p}:{Xmean}")
        if any(k == p for k in ["precision", "recall", "f1", "support"]):
            if p == "support":
                weights = np.mean(X, axis=0)
                X = X / sum(Xmean)
                Xmean = np.mean(X, axis=0)
                Xstd = np.std(X, axis=0)
            Y[0].append(Xmean[0])
            Y[1].append(Xmean[1])
            Yerr[0].append(Xstd[0])
            Yerr[1].append(Xstd[1])
            Xlabel.append(p)

    plt.figure(figsize=(7, 5))
    pos = [0, 0.35, 1, 1.35, 2, 2.35, 2.70, 2.70+1-0.5, 2.70+1-0.35+0.35]
    pos = np.arange(len(Y[1]))
    plt.bar(pos, Y[0], 0.35, yerr=Yerr[0], label=target[0])

    pos = np.arange(len(Y[1])) + 0.35
    plt.bar(pos, Y[1], 0.35, yerr=Yerr[1], label=target[1])

    pos = (np.arange(len(Y[1])) + 0.35/2)
    plt.bar(pos, np.average([Y[0], Y[1]], axis=0, weights=weights), 0.10,
            # yerr=np.average([Yerr[0], Yerr[1]], axis=0, weights=weights),
            label="Weighted average", color="red")
    plt.bar(pos, np.average([Y[0], Y[1]], axis=0), 0.10,
            # yerr=np.average([Yerr[0], Yerr[1]], axis=0),
            label="Average", color="green")
    plt.xticks(np.arange(len(Xlabel)) + 0.35/2, Xlabel)
    plt.legend(loc="upper left")
# %%
plotresult(ldict)
# %%
plotresult([ldict[i] for i in mu["muw0.4"]])
# %%
plotresult([ldict[i] for i in mu["muw0.3"]])
# %%
plotresult([ldict[i] for i in mu["muw0.2"]])
