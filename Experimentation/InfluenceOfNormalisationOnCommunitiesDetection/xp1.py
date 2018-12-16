import numpy as np
import matplotlib.pyplot as plt
import re
import sys
sys.path.append("..//Toolbox")
from Utils import architecture
# %%
ldict, list_graph, dmk, dk, dmu, reslabel = architecture("xp1.pickle")
print(reslabel)
# %% Average performance function of meta-param
paramkeys = [("allparam", list_graph),
             # ("muw0.4", dmu["muw0.4"]),
             # ("muw0.3", dmu["muw0.3"]),
             # ("muw0.2", dmu["muw0.2"])
             ]
for keyname, paramkey in paramkeys:
    ARIClassement = sorted([dict(
        method=k,
        mean=np.mean([ 1 - ldict[path][k] for path in paramkey ]),
        std=np.std([ 1 - ldict[path][k] for path in paramkey ]),
        samples=len([ 1 - ldict[path][k] for path in paramkey ]))
                            for k in reslabel if "ARI" in k], key=lambda x: x["mean"],
                            reverse=True)
    NMIClassement = sorted([dict(
        method=k,
        mean=np.mean([ 1 - ldict[path][k] for path in paramkey ]),
        std=np.std([ 1 - ldict[path][k] for path in paramkey ]),
        samples=len([ 1 - ldict[path][k] for path in paramkey ]))
                            for k in reslabel if "NMI" in k], key=lambda x: x["mean"],
                            reverse=True)
    print("ARI classement ", keyname, ":")
    for i, d in enumerate(ARIClassement):
        print(f"{i + 1}: {d}")
    print("NMI classement ", keyname, ":")
    for i, d in enumerate(NMIClassement):
        print(f"{i + 1}: {d}")
# %%
cmap = plt.cm.get_cmap('tab10').colors
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(15, 10))
muKeys = sorted(list(dmu.keys()))
dataFilter = lambda paths, key: [ 1 - ldict[path][key] for path in paths ]
ax0.set_ylabel("Similarity NMI")
Ymean = np.array([np.mean(dataFilter(dmu[curmu], 'NMI_Louvain_')) for curmu in muKeys])
Ystd = np.array([np.std(dataFilter(dmu[curmu], 'NMI_Louvain_')) for curmu in muKeys])
color = cmap[0]
ax0.plot(muKeys, Ymean, label='Louvain', linewidth=4, color=color)
# ax0.plot(Xmu, Ymean - Ystd, linestyle="--", linewidth=0.5, color=color)
# ax0.plot(Xmu, Ymean + Ystd, linestyle="--", linewidth=0.5, color=color)
Ymean = np.array([np.mean(dataFilter(dmu[curmu], 'NMI_PLP_')) for curmu in muKeys])
Ystd = np.array([np.std(dataFilter(dmu[curmu], 'NMI_PLP_')) for curmu in muKeys])
color = cmap[1]
ax0.plot(muKeys, Ymean, label='PLP', linewidth=4, color=color)
# ax0.plot(Xmu, Ymean - Ystd, linestyle="--", linewidth=0.5, color=color)
# ax0.plot(Xmu, Ymean + Ystd, linestyle="--", linewidth=0.5, color=color)

ax1.set_ylabel("Similarity ARI")
Ymean = np.array([np.mean(dataFilter(dmu[curmu], 'ARI_Louvain_')) for curmu in muKeys])
Ystd = np.array([np.std(dataFilter(dmu[curmu], 'ARI_Louvain_')) for curmu in muKeys])
color = cmap[0]
ax1.plot(muKeys, Ymean, label='Louvain', linewidth=4, color=color)
# ax1.plot(Xmu, Ymean - Ystd, linestyle="--", linewidth=0.5, color=color)
# ax1.plot(Xmu, Ymean + Ystd, linestyle="--", linewidth=0.5, color=color)
Ymean = np.array([np.mean(dataFilter(dmu[curmu], 'ARI_PLP_')) for curmu in muKeys])
Ystd = np.array([np.std(dataFilter(dmu[curmu], 'ARI_PLP_')) for curmu in muKeys])
color = cmap[1]
ax1.plot(muKeys, Ymean, label='PLP', linewidth=4, color=color)
# ax1.plot(Xmu, Ymean - Ystd, linestyle="--", linewidth=0.5, color=color)
# ax1.plot(Xmu, Ymean + Ystd, linestyle="--", linewidth=0.5, color=color)


for i, param in enumerate(filter(lambda x: "glove" in x and "NMI" in x, reslabel)):
    color = cmap[i + 2]
    Ymean = np.array([np.mean(dataFilter(dmu[curmu], param)) for curmu in muKeys])
    Ystd = np.array([np.std(dataFilter(dmu[curmu], param)) for curmu in muKeys])
    ax0.plot(muKeys, Ymean, label=param[4:], linewidth=2, color=color)
    # ax0.plot(Xmu, Ymean - Ystd, linestyle="--", linewidth=0.5, color=color)
    # ax0.plot(Xmu, Ymean + Ystd, linestyle="--", linewidth=0.5, color=color)
for i, param in enumerate(filter(lambda x: "glove" in x and "ARI" in x, reslabel)):
    color = cmap[i + 2]
    Ymean = np.array([np.mean(dataFilter(dmu[curmu], param)) for curmu in muKeys])
    Ystd = np.array([np.std(dataFilter(dmu[curmu], param)) for curmu in muKeys])
    algo, xmax, alpha = re.match(r"^(?:(?:NMI|ARI)_)?(Louvain|PLP)_glove(?:xmax(\d+)alpha(\d+\.\d+))?$", param).groups()
    param = f"Glove[xmax={xmax}; alpha={alpha}]+{algo}" if xmax is not None else f"Glove[xmax=100; alpha=0.75]+{algo}"
    ax1.plot(muKeys, Ymean, label=param, linewidth=2, color=color)
    # ax1.plot(Xmu, Ymean - Ystd, linestyle="--", linewidth=0.5, color=color)
    # ax1.plot(Xmu, Ymean + Ystd, linestyle="--", linewidth=0.5, color=color)

ax0.set_xticks(muKeys)
ax1.set_xticks(muKeys)
ax0.legend()
plt.savefig(f"InfluenceOfGloveOnCommunityDetection.pdf")
# %%
# plt.savefig(f"GloveMuImpact.pdf")
