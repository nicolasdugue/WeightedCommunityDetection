# import itertools
import sys
sys.path.append("../Toolbox")
import os
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
from Utils import architecture
# %%
for xpname in ["xp3_7", "xp3"]:
    # pattern = re.compile(r".*reference_model_7/mk(\d+)k(\d+)muw(\d+(?:\.\d+)?)\.model(\d+)\.dat$")
    ldict, list_graph, mk, k, mu, reslabel = architecture(f"{xpname}.pickle")
    for paramtest in mu:
        Y = {kmu: [[], []] for kmu in mu}
        Yerr = {kmu: [[], []] for kmu in mu}
        Xlabel = []
        weights = {}
        ldictfilt = list(filter(lambda x: x[1] is not None and x[0] in mu[paramtest] and None not in x[1].values(), enumerate(ldict)))
        target = list(ldictfilt[0][1].values())[0]["target"]
        features = list(ldictfilt[0][1].values())[0]["features"]
        reslabel = list(ldictfilt[0][1][next(x for x in ldictfilt[0][1])].keys())
        print("target:", target)
        print("features:", features)
        for p in reslabel:
            if p != "target" and p != "features":
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
                if p == "support":
                    for kmu in mu:
                        weights[kmu] = np.mean(X[kmu], axis=0)
                        X[kmu] = X[kmu] / sum(Xmean[kmu])
                        Xmean[kmu] = np.mean(X[kmu], axis=0)
                        Xstd[kmu] = np.std(X[kmu], axis=0)
                if p != "support":
                    Xlabel.append(p)
                    for kmu in mu:
                            Y[kmu][0].append(Xmean[kmu][0])
                            Y[kmu][1].append(Xmean[kmu][1])
                            Yerr[kmu][0].append(Xstd[kmu][0])
                            Yerr[kmu][1].append(Xstd[kmu][1])

        fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(15, 10))
        plt.suptitle(f"Result on Graph with {paramtest}")
        plt.subplots_adjust(hspace=0.4)
        for i, kmu in enumerate(mu):
            last_patches = list()

            ax[i].bar(np.arange(len(Y[kmu][0])), Y[kmu][0], 0.35, yerr=Yerr[kmu][0], label=target[0], color="blue")
            for rect in filter(lambda x: x not in last_patches, ax[i].patches):
                height = rect.get_height()
                ax[i].text(rect.get_x() + rect.get_width()/3, -0.1, f"{height:.2f}", color="blue")
            last_patches = list(ax[i].patches)

            ax[i].bar(np.arange(len(Y[kmu][1])) + 0.35, Y[kmu][1], 0.35, yerr=Yerr[kmu][1], label=target[1], color="orange")
            for rect in filter(lambda x: x not in last_patches, ax[i].patches):
                height = rect.get_height()
                ax[i].text(rect.get_x() + rect.get_width()/3, -0.1, f"{height:.2f}", color="orange")
            last_patches = list(ax[i].patches)

            ax[i].bar(np.arange(len(Y[kmu][1])) + 0.35/2, np.average([Y[kmu][0], Y[kmu][1]], axis=0, weights=weights[kmu]), 0.10,
                    # yerr=np.average([Yerr[0], Yerr[1]], axis=0, weights=weights),
                        label="Weighted average", color="red")
            for rect in filter(lambda x: x not in last_patches, ax[i].patches):
                height = rect.get_height()
                ax[i].text(rect.get_x(), -0.08, f"{height:.2f}", color="red")
            last_patches = list(ax[i].patches)

            # ax[i].bar(np.arange(len(Y[kmu][1])) + 0.35/2, np.average([Y[kmu][0], Y[kmu][1]], axis=0), 0.10,
            #         # yerr=np.average([Yerr[0], Yerr[1]], axis=0),
            #         label="Average", color="green")
            # for rect in filter(lambda x: x not in last_patches, ax[i].patches):
            #     height = rect.get_height()
            #     ax[i].text(rect.get_x(), -0.16, f"{height:.2f}", color="green")
            for rect, x in zip(filter(lambda x: x not in last_patches, ax[i].patches), Xlabel):
                ax[i].text(rect.get_x(), -0.3, x)
            last_patches = list(ax[i].patches)

            ax[i].set_title(f"{xpname} Training Graph {kmu}")
            ax[i].set_yticks(np.arange(0, 1.1, 0.1))
        plt.xticks([], [])
        plt.legend(loc="upper left")
        plt.savefig(f"plot/{xpname}_Test={paramtest}_2.pdf")
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
