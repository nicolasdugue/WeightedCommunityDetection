# import itertools
import os
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
# %%

list_graph = []
mk = {}
k = {}
mu = {}
pattern = re.compile(r".*reference_model_7/mk(\d+)k(\d+)muw(\d+(?:\.\d+)?)\.model(\d+)\.dat$")
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
        with open(os.path.join(path, "xp3_7.pickle"), "rb") as file:
            reslabel = list(list(pickle.load(file).values())[0].keys())
    except FileNotFoundError as e:
        # print(e)
        continue
    break
# print(reslabel)
# %%
failled, ldict = 0, []
for path in list_graph:
    try:
        with open(os.path.join(path, "xp3_7.pickle"), "rb") as file:
            ldict.append(pickle.load(file))
    except FileNotFoundError as e:
        failled += 1
        ldict.append(None)
        # print(e)
        continue
print(f"fail: {failled}/{len(list_graph)}")
# %%
# Y = [[], []]
# Yerr = [[], []]
# target = list(ldict[0].values())[0]["target"]
# Xlabel = []
# for p in reslabel:
#     if p == "target" or p == "features":
#         print(f"{p}:{list(ldict[0].values())[0][p]}")
#     else:
#         X = []
#         for drefs in ldict:
#             if drefs is not None:
#                 for dref in drefs.values():
#                     X.append(dref[p])
#         Xmean = np.mean(X, axis=0)
#         Xstd = np.std(X, axis=0)
#         print(f"{p}:{Xmean}")
#     if any(k == p for k in ["precision", "recall", "f1", "support"]):
#         if p == "support":
#             X = X / sum(Xmean)
#             Xmean = np.mean(X, axis=0)
#             Xstd = np.std(X, axis=0)
#         Y[0].append(Xmean[0])
#         Y[1].append(Xmean[1])
#         Yerr[0].append(Xstd[0])
#         Yerr[1].append(Xstd[1])
#         Xlabel.append(p)
# plt.bar(np.arange(len(Y[0])), Y[0], 0.35, yerr=Yerr[0], label=target[0])
# plt.bar(np.arange(len(Y[1])) + 0.35, Y[1], 0.35, yerr=Yerr[1], label=target[1])
# plt.xticks(np.arange(len(Xlabel)) + 0.35/2, Xlabel)
# plt.legend()

# %%
Y = {kmu: [[], []] for kmu in mu}
Yerr = {kmu: [[], []] for kmu in mu}
target = list(ldict[0].values())[0]["target"]
Xlabel = []
weights = {}
ldictfilt = list(filter(lambda x: x[1] is not None and x[0] in mu["muw0.4"], enumerate(ldict)))
for p in reslabel:
    if p == "target" or p == "features":
        print(f"{p}:{list(ldict[0].values())[0][p]}")
    else:
        X = {kmu: [] for kmu in mu}
        Xmean = {}
        Xstd = {}
        for i, drefs in ldictfilt:
            for path, dref in drefs.items():
                for kmu in mu:
                    if kmu in path:
                        X[kmu].append(dref[p])
        # print(X)
        for kmu in mu:
            Xmean[kmu] = np.mean(X[kmu], axis=0)
            Xstd[kmu] = np.std(X[kmu], axis=0)
            print(f"{p}({kmu}):{Xmean[kmu]}")
    if any(k == p for k in ["precision", "recall", "f1", "support"]):
        Xlabel.append(p)
        if p == "support":
            for kmu in mu:
                weights[kmu] = np.mean(X[kmu], axis=0)
                X[kmu] = X[kmu] / sum(Xmean[kmu])
                Xmean[kmu] = np.mean(X[kmu], axis=0)
                Xstd[kmu] = np.std(X[kmu], axis=0)
        for kmu in mu:
                Y[kmu][0].append(Xmean[kmu][0])
                Y[kmu][1].append(Xmean[kmu][1])
                Yerr[kmu][0].append(Xstd[kmu][0])
                Yerr[kmu][1].append(Xstd[kmu][1])

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(15, 10))
for i, kmu in enumerate(mu):
    ax[i].bar(np.arange(len(Y[kmu][0])), Y[kmu][0], 0.35, yerr=Yerr[kmu][0], label=target[0])
    ax[i].bar(np.arange(len(Y[kmu][1])) + 0.35, Y[kmu][1], 0.35, yerr=Yerr[kmu][1], label=target[1])
    ax[i].bar(np.arange(len(Y[kmu][1])) + 0.35/2, np.average([Y[kmu][0], Y[kmu][1]], axis=0, weights=weights[kmu]), 0.10,
            # yerr=np.average([Yerr[0], Yerr[1]], axis=0, weights=weights),
            label="Weighted average", color="red")
    ax[i].bar(np.arange(len(Y[kmu][1])) + 0.35/2, np.average([Y[kmu][0], Y[kmu][1]], axis=0), 0.10,
            # yerr=np.average([Yerr[0], Yerr[1]], axis=0),
            label="Average", color="green")
plt.xticks(np.arange(len(Xlabel)) + 0.35/2, Xlabel)
plt.legend()
# %%
