import itertools
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../Toolbox")
from Utils import architecture
# %%
ldict, _, _, mu, reslabel = architecture("xp4.pickle")
ldict7, _, _, _, reslabel7 = architecture("xp4_7.pickle")
print(reslabel7)
# %%
def plotRes(ldict, param, title, removedCurves=[]):
    cmap = itertools.cycle(plt.cm.get_cmap('tab10').colors)
    XIndex = sorted(list(mu.keys()))
    plt.figure(figsize=(10,5))
    for curbname in filter(lambda x: param in x and x not in removedCurves, reslabel):
        Y, Yerr = [], []
        for label in XIndex:
            if "train" in curbname:
                X0 = [np.mean([1 - x for x in ldict[path][curbname]], axis=0) for path in filter(lambda x: label in x, ldict)]
                X1 = [np.std([1 - x for x in ldict[path][curbname]], axis=0) for path in filter(lambda x: label in x, ldict)]
                Xmean = np.mean(X0, axis=0)
                Xstd = np.mean(X1, axis=0)
            else:
                X = [ 1 - ldict[path][curbname] for path in filter(lambda x: label in x, ldict)]
                Xmean = np.mean(X, axis=0)
                Xstd = np.mean(X, axis=0)
            Y.append(Xmean)
            Yerr.append(Xstd)
        plt.plot(XIndex, Y, color=next(cmap), label=curbname)
    plt.title(title)
    plt.ylabel(param + " Similarity")
    plt.legend()
# %%
plotRes(ldict, param="NMI",
        title=f"Comparaison of ImprovePLP(5features) and other methods in term of NMI similarity",
        removedCurves=[ 'train=k20_NMI_PLPnico_',
                        'train=k15_NMI_PLPnico_',
                        'train=mk500_NMI_PLPnico_',
                        'train=mk100_NMI_PLPnico_',
                        'train=mk300_NMI_PLPnico_',
                        'NMI_PLP_glovexmax30alpha0.3',
                        'train=k25_NMI_PLPnico_'])
plotRes(ldict7, param="NMI",
        title=f"Comparaison of ImprovePLP(7features) and other methods in term of NMI similarity",
        removedCurves=[ 'train=k20_NMI_PLPnico_',
                        'train=k15_NMI_PLPnico_',
                        'train=mk500_NMI_PLPnico_',
                        'train=mk100_NMI_PLPnico_',
                        'train=mk300_NMI_PLPnico_',
                        'NMI_PLP_glovexmax30alpha0.3',
                        'train=k25_NMI_PLPnico_'])
plotRes(ldict, param="ARI",
        title=f"Comparaison of ImprovePLP(5features) and other methods in term of ARI similarity",
        removedCurves=[ 'train=k20_ARI_PLPnico_',
                        'train=k15_ARI_PLPnico_',
                        'train=mk500_ARI_PLPnico_',
                        'train=mk100_ARI_PLPnico_',
                        'train=mk300_ARI_PLPnico_',
                        'ARI_PLP_glovexmax30alpha0.3',
                        'train=k25_ARI_PLPnico_'])
plotRes(ldict7, param="ARI",
        title=f"Comparaison of ImprovePLP(7features) and other methods in term of ARI similarity",
        removedCurves=[ 'train=k20_ARI_PLPnico_',
                        'train=k15_ARI_PLPnico_',
                        'train=mk500_ARI_PLPnico_',
                        'train=mk100_ARI_PLPnico_',
                        'train=mk300_ARI_PLPnico_',
                        'ARI_PLP_glovexmax30alpha0.3',
                        'train=k25_ARI_PLPnico_'])
