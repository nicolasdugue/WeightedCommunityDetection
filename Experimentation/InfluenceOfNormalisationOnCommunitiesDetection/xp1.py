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
for path in reversed(list_graph):
    try:
        with open(os.path.join(path, "xp1.pickle"), "rb") as file:
            reslabel = list(pickle.load(file).keys())
    except FileNotFoundError as e:
        # print(e)
        continue
    break
print("__Result__")
# print(reslabel)
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
        try:
            if "NMI" in p:
                matres[gid][i] = 1 - resg[p]
            else:
                matres[gid][i] = resg[p]
        except KeyError:
            matres[gid][i] = None
        i += 1
print(f"fail: {failled}/{len(list_graph)}")

# %% Average performance function of meta-param
paramkeys = [("allparam", slice(0, len(list_graph))), ("muw0.4", mu["muw0.4"]), ("muw0.3", mu["muw0.3"]), ("muw0.2", mu["muw0.2"])]
for keyname, paramkey in paramkeys:
    ARIClassement = sorted([dict(
        method=k,
        mean=np.mean(matres[paramkey, reslabel.index(k) + 3][~np.isnan(matres[paramkey, reslabel.index(k) + 3])]),
        std=np.std(matres[paramkey, reslabel.index(k) + 3][~np.isnan(matres[paramkey, reslabel.index(k) + 3])]),
        samples=len(matres[paramkey, reslabel.index(k) + 3][~np.isnan(matres[paramkey, reslabel.index(k) + 3])]))
                            for k in reslabel if "ARI" in k], key=lambda x: x["mean"])
    NMIClassement = sorted([dict(
        method=k,
        mean=np.mean(matres[paramkey, reslabel.index(k) + 3][~np.isnan(matres[paramkey, reslabel.index(k) + 3])]),
        std=np.std(matres[paramkey, reslabel.index(k) + 3][~np.isnan(matres[paramkey, reslabel.index(k) + 3])]),
        samples=len(matres[paramkey, reslabel.index(k) + 3][~np.isnan(matres[paramkey, reslabel.index(k) + 3])]))
                            for k in reslabel if "NMI" in k], key=lambda x: x["mean"],
                           reverse=True)
    print("ARI classement ", keyname, ":")
    for i, d in enumerate(ARIClassement[0:5]):
        print(i + 1, ": ", d)
    print("NMI classement ", keyname, ":")
    for i, d in enumerate(NMIClassement[0:5]):
        print(i + 1, ": ", d)

# %%


def dataFilter(listIndex, param):
    tab = matres[listIndex, reslabel.index(param) + 3]
    return tab[~np.isnan(tab)]


# %%
cmap = plt.cm.get_cmap('tab10').colors
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(15, 10))
Xmu = [0.2, 0.3, 0.4]
muKeys = list(mu.keys())

ax0.set_ylabel("Similarity NMI")
Ymean = np.array([np.mean(dataFilter(mu[curmu], 'NMILouvain')) for curmu in muKeys])
Ystd = np.array([np.std(dataFilter(mu[curmu], 'NMILouvain')) for curmu in muKeys])
color = cmap[0]
ax0.plot(Xmu, Ymean, label='Louvain', linewidth=4, color=color)
# ax0.plot(Xmu, Ymean - Ystd, linestyle="--", linewidth=0.5, color=color)
# ax0.plot(Xmu, Ymean + Ystd, linestyle="--", linewidth=0.5, color=color)
Ymean = np.array([np.mean(dataFilter(mu[curmu], 'NMIPLP')) for curmu in muKeys])
Ystd = np.array([np.std(dataFilter(mu[curmu], 'NMIPLP')) for curmu in muKeys])
color = cmap[1]
ax0.plot(Xmu, Ymean, label='PLP', linewidth=4, color=color)
# ax0.plot(Xmu, Ymean - Ystd, linestyle="--", linewidth=0.5, color=color)
# ax0.plot(Xmu, Ymean + Ystd, linestyle="--", linewidth=0.5, color=color)

ax1.set_ylabel("Dissimilarity ARI")
Ymean = np.array([np.mean(dataFilter(mu[curmu], 'ARILouvain')) for curmu in muKeys])
Ystd = np.array([np.std(dataFilter(mu[curmu], 'ARILouvain')) for curmu in muKeys])
color = cmap[0]
ax1.plot(Xmu, Ymean, label='Louvain', linewidth=4, color=color)
# ax1.plot(Xmu, Ymean - Ystd, linestyle="--", linewidth=0.5, color=color)
# ax1.plot(Xmu, Ymean + Ystd, linestyle="--", linewidth=0.5, color=color)
Ymean = np.array([np.mean(dataFilter(mu[curmu], 'ARIPLP')) for curmu in muKeys])
Ystd = np.array([np.std(dataFilter(mu[curmu], 'ARIPLP')) for curmu in muKeys])
color = cmap[1]
ax1.plot(Xmu, Ymean, label='PLP', linewidth=4, color=color)
# ax1.plot(Xmu, Ymean - Ystd, linestyle="--", linewidth=0.5, color=color)
# ax1.plot(Xmu, Ymean + Ystd, linestyle="--", linewidth=0.5, color=color)


for i, param in enumerate(filter(lambda x: "NMIglove" in x, reslabel)):
    color = cmap[i + 2]
    Ymean = np.array([np.mean(dataFilter(mu[curmu], param)) for curmu in muKeys])
    Ystd = np.array([np.std(dataFilter(mu[curmu], param)) for curmu in muKeys])
    ax0.plot(Xmu, Ymean, label=param[3:], linewidth=2, color=color)
    # ax0.plot(Xmu, Ymean - Ystd, linestyle="--", linewidth=0.5, color=color)
    # ax0.plot(Xmu, Ymean + Ystd, linestyle="--", linewidth=0.5, color=color)
for i, param in enumerate(filter(lambda x: "ARIglove" in x, reslabel)):
    color = cmap[i + 2]
    Ymean = np.array([np.mean(dataFilter(mu[curmu], param)) for curmu in muKeys])
    Ystd = np.array([np.std(dataFilter(mu[curmu], param)) for curmu in muKeys])
    ax1.plot(Xmu, Ymean, label=param[3:], linewidth=2, color=color)
    # ax1.plot(Xmu, Ymean - Ystd, linestyle="--", linewidth=0.5, color=color)
    # ax1.plot(Xmu, Ymean + Ystd, linestyle="--", linewidth=0.5, color=color)

plt.xticks(Xmu)
plt.legend(bbox_to_anchor=(0.25, 1.75), loc=1, borderaxespad=0)
plt.savefig(f"GloveMuImpact.pdf")
