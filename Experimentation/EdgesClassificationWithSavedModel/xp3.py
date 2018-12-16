import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../Toolbox")
from Utils import architecture
# %%
ldict, _, _, mu, reslabel = architecture("xp3.pickle")
ldict7, _, _, _, reslabel7 = architecture("xp3_7.pickle")
print(reslabel7)
# %%
def plotRes(ldict, test, param):
    target = ldict[next(k for k in ldict.keys())]["target"]
    Xlabel = ["support", "precision", "recall", "f1"]
    Y = [[], []]
    Yerr = [[], []]
    for label in Xlabel:
        X0 = [np.mean(ldict[path][param + "_" + label], axis=0) for path in filter(lambda x: test in x, ldict)]
        X1 = [np.std(ldict[path][param + "_" + label], axis=0) for path in filter(lambda x: test in x, ldict)]
        Xmean = np.mean(X0, axis=0)
        Xstd = np.mean(X1, axis=0)
        if label == "support":
            weights = np.mean(X0, axis=0)
        else:
            Y[0].append(Xmean[0])
            Y[1].append(Xmean[1])
            Yerr[0].append(Xstd[0])
            Yerr[1].append(Xstd[1])
    del Xlabel[0]
    plt.figure(figsize=(7, 5))
    pos = np.arange(len(Y[1]))
    ax = plt.bar(pos, Y[0], 0.35, yerr=Yerr[0], label=target[0])
    for rect in ax.patches:
        plt.text(rect.get_x() + rect.get_width()/4, -0.1, f"{rect.get_height():.2f}", color="blue")

    pos = np.arange(len(Y[1])) + 0.35
    ax = plt.bar(pos, Y[1], 0.35, yerr=Yerr[1], label=target[1])
    for rect in ax.patches:
        plt.text(rect.get_x() + rect.get_width()/4, -0.1, f"{rect.get_height():.2f}", color="orange")


    pos = (np.arange(len(Y[1])) + 0.35/2)
    ax = plt.bar(pos, np.average([Y[0], Y[1]], axis=0, weights=weights), 0.10,
            # yerr=np.average([Yerr[0], Yerr[1]], axis=0, weights=weights),
            label="Weighted average", color="red")
    for rect in ax.patches:
        plt.text(rect.get_x() - rect.get_width()/2, -0.05, f"{rect.get_height():.2f}", color="red")

    # plt.bar(pos, np.average([Y[0], Y[1]], axis=0), 0.10,
    #         # yerr=np.average([Yerr[0], Yerr[1]], axis=0),
    #         label="Average", color="green")
    print(Xlabel)
    for rect, x in zip(ax.patches, Xlabel):
        plt.text(rect.get_x() - len(x)/100 , -0.14, x)
    plt.title(f"Result of model {param} in graph with {test}")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks([], [])
    plt.legend(loc="upper left")
    plt.show()
# %%
plotRes(ldict, "muw0.4", "train=muw0.4")
