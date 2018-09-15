import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../Toolbox")
from Utils import architecture
# %%
def plotresult(ldict, reslabel, title=""):
    Y = [[], []]
    Yerr = [[], []]
    target = ldict[next(k for k in ldict.keys())]["target"]
    Xlabel = []
    for p in reslabel:
        if p == "target" or p == "features":
            print(f"{p}:{ldict[next(k for k in ldict.keys())][p]}")
        else:
            X = [ldict[path][p] for path in ldict if ldict[path]]
            Xmean = np.mean(X, axis=0)
            Xstd = np.std(X, axis=0)
            print(f"{p}:{Xmean}")
        if any(k == p for k in ["precision", "recall", "f1", "support"]):
            if p == "support":
                weights = np.mean(X, axis=0)
                X = X / sum(Xmean)
                Xmean = np.mean(X, axis=0)
                Xstd = np.std(X, axis=0)
            else:
                Xlabel.append(p)
                Y[0].append(Xmean[0])
                Y[1].append(Xmean[1])
                Yerr[0].append(Xstd[0])
                Yerr[1].append(Xstd[1])

    plt.figure(figsize=(15, 10))
    pos = np.arange(len(Y[1]))
    ax = plt.bar(pos, Y[0], 0.35, yerr=Yerr[0], label=target[0])
    for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, -0.1, f"{height:.2f}", color="blue")

    pos = np.arange(len(Y[1])) + 0.35
    ax = plt.bar(pos, Y[1], 0.35, yerr=Yerr[1], label=target[1])
    for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, -0.1, f"{height:.2f}", color="orange")


    pos = (np.arange(len(Y[1])) + 0.35/2)
    ax = plt.bar(pos, np.average([Y[0], Y[1]], axis=0, weights=weights), 0.10,
            # yerr=np.average([Yerr[0], Yerr[1]], axis=0, weights=weights),
            label="Weighted average", color="red")
    for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, -0.05, f"{height:.2f}", color="red")

    # plt.bar(pos, np.average([Y[0], Y[1]], axis=0), 0.10,
    #         # yerr=np.average([Yerr[0], Yerr[1]], axis=0),
    #         label="Average", color="green")
    print(Xlabel)
    for rect, x in zip(ax.patches, Xlabel):
        plt.text(rect.get_x(), -0.18, x)
    plt.title(title)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks([], [])
    plt.legend(loc="upper left")
    plt.show()
# %%
ldict7, _, _, mu7, reslabel7 = architecture("xp2_7.pickle")
print(reslabel7)
# %%
plotresult(ldict7, reslabel7)
# %%
plotresult({i: ldict7[i] for i in mu7["muw0.4"]}, reslabel7)
# %%
plotresult({i: ldict7[i] for i in mu7["muw0.3"]}, reslabel7)
# %%
plotresult({i: ldict7[i] for i in mu7["muw0.2"]}, reslabel7)
# %%
ldict,  _, _, mu, reslabel = architecture("xp2.pickle")
print(reslabel)
# %%
plotresult(ldict, reslabel)
# %%
plotresult({i: ldict[i] for i in mu["muw0.4"]}, reslabel)
# %%
plotresult({i: ldict[i] for i in mu["muw0.3"]}, reslabel)
# %%
plotresult({i: ldict[i] for i in mu["muw0.2"]}, reslabel)
# %%
