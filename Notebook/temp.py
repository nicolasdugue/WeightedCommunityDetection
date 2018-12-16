import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("Experimentation/Toolbox")
from Utils import loadings, representativeSamples, extractParams
# %%
ref = representativeSamples()
res = {}
target = ["outside", "inside"]
color = ["red", "blue"]
Xmean, Xerr = {}, {}
for i, path in enumerate(ref):
    G, gt_partition, res = loadings(path, verbose=False)
    _, _, mu = extractParams(path)
    inside, outside = [], []
    for u, v in G.edges():
        if gt_partition.subsetOf(u) == gt_partition.subsetOf(v):
            inside.append(G.weight(u, v))
        else:
            outside.append(G.weight(u, v))
    try:
        Xmean[mu].append(np.array([np.mean(outside), np.mean(inside)]))
        Xerr[mu].append(np.array([np.std(outside), np.std(inside)]))
    except KeyError:
        Xmean[mu] = [np.array([np.mean(outside), np.mean(inside)])]
        Xerr[mu] = [np.array([np.std(outside), np.std(inside)])]
    if i % 10 == 0 or i == len(ref) - 1:
        print(f"[{i+1}/{len(ref)}]: {(i+1)/len(ref)*100:.2f} % {path}")
# %%
Xlabel = sorted(list(Xmean.keys()))
for mu in Xlabel:
    Xerr[mu] = np.std(Xmean[mu], axis=0)
    Xmean[mu] = np.mean(Xmean[mu], axis=0)
    # Xerr[mu] = np.mean(Xerr[mu], axis=0)
# %%
for i in range(2):
    plt.plot(Xlabel, [Xmean[mu][i] for mu in Xlabel], label=f"{target[i]} links", linewidth=2, color=color[i])
    plt.plot(Xlabel, [Xmean[mu][i] + Xerr[mu][i] for mu in Xlabel],  color=color[i], linewidth=0.5, linestyle="--")
    plt.plot(Xlabel, [Xmean[mu][i] - Xerr[mu][i] for mu in Xlabel], label=f"Bounds of standard deviation",  color=color[i], linewidth=0.5, linestyle="--")
plt.legend()
plt.ylabel("Average Weights")
