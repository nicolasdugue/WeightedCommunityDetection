import itertools
import sys
import re
sys.path.append("../Toolbox")
import matplotlib.pyplot as plt
import numpy as np
from Utils import architecture
# %%
pattern = re.compile(r".*reference_model(?:_7)?/(mk\d+)(k\d+)(muw\d+(?:\.\d+)?)\.model(?:\d+)\.dat$")
# %%
for xpname in ["xp4_7", "xp4"]:
    # %%
    xpname = "xp4_7"
    ldict, list_graph, mk, k, mu, resLabel = architecture(f"{xpname}.pickle")
    # %%
    res = {}
    genRes = {}
    resLabel = list(ldict[3].keys())
    labResModel = ldict[0][next(k for k in ldict[0].keys() if "reference_model" in k)]
    for i, path in enumerate(list_graph):
        if ldict[i] is not None:
            res[path] = {resLab: {k:[] for k in itertools.chain(mk, k , mu, ["all"])} for resLab in labResModel}
            genRes[path] = {}
            for lab in resLabel:
                if "reference_model" in lab:
                    curmk, curk , curmu = pattern.match(lab).groups()
                    for resLab in labResModel:
                        try:
                            res[path][resLab].get(curmk, []).append(ldict[i][lab][resLab])
                            res[path][resLab].get(curk, []).append(ldict[i][lab][resLab])
                            res[path][resLab].get(curmu, []).append(ldict[i][lab][resLab])
                            res[path][resLab].get("all", []).append(ldict[i][lab][resLab])
                        except KeyError as e:
                            # print(e)
                            # print("_____ERROR_____")
                            # print(path)
                            # print(resLab)
                            # print("curmk :", curmk, " curk :", curk, " curmu :", curmu)
                            break
                elif lab == "own":
                    for sk in ldict[i][lab]:
                        genRes[path]["own" + sk] = ldict[i][lab][sk]
                else:
                    try:
                        genRes[path][lab] = ldict[i][lab]
                    except KeyError as e:
                        # print(e)
                        break
        if i % 10 == 0 or i == len(list_graph) - 1 :
            pass
            # print(f"[{i+1}/{len(list_graph)}]progress {(i+1)/len(list_graph)*100:.2f}%")
    # %% sort by testparam
    dictRes = {}
    for i, path in enumerate(res):
        dictRes[path] = {}
        for resLab, paramD in genRes[path].items():
            dictRes[path][resLab] = paramD
        for resLab, paramD in res[path].items():
            if bool(paramD["all"]):
                if not all(paramD):
                    print(paramD)
                    break
                if resLab == "features" or resLab == "target":
                    dictRes[path][resLab] = paramD['all'][0]
                    target = paramD['all'][0]
                    features = paramD['all'][0]
                else:
                    dictRes[path][resLab] = {param :{"mean": np.mean(res[path][resLab][param], axis=0),
                                                     "std": np.std(res[path][resLab][param], axis=0),
                                                     "samples": len(res[path][resLab][param])} for param in paramD}
        if i % 10 == 0 or i == len(res) - 1 :
            print(f"[{i+1}/{len(res)}]progress {(i+1)/len(res)*100:.2f}%")
# %%
curbs = {}
cmap = itertools.cycle(plt.cm.get_cmap('tab10').colors)
# Xindex for each test parameter
XIndex = sorted(list(mu.keys()))
mesure = "ARI"
graphIndex = [list(filter(lambda key: kmu in key, dictRes)) for kmu in XIndex]
curbsname = list(filter(lambda key: mesure in key and not "glove" in key, dictRes['../../lfr_5000/mk500/k25/muw0.4/5']))
print("target:", target)
print("features:", features)
plt.figure(figsize=(10,5))
for curbname in curbsname:
    if curbname == "ARI_PLPnico_" or curbname == "NMI_PLPnico_":
        Y, Yerr = {trainmu: [] for trainmu in mu}, {trainmu: [] for trainmu in mu}
        for listPath in graphIndex:
            for trainmu in mu:
                if mesure == "NMI":
                    Y[trainmu].append(1-np.mean([dictRes[path][curbname][trainmu]["mean"] for path in listPath if curbname in dictRes[path] and not np.isnan(dictRes[path][curbname][trainmu]["mean"])]))
                    # Yerr[trainmu].append(1-np.mean([dictRes[path][curbname][trainmu]["std"] for path in listPath if curbname in dictRes[path]]))
                else:
                    Y[trainmu].append( np.mean([dictRes[path][curbname][trainmu]["mean"] for path in listPath if curbname in dictRes[path] and not np.isnan(dictRes[path][curbname][trainmu]["mean"])]))
                    # Yerr[trainmu].append(np.mean([dictRes[path][curbname][trainmu]["std"] for path in listPath if curbname in dictRes[path]]))
        for trainmu in mu:
            plt.plot(XIndex, Y[trainmu], color=next(cmap), label=curbname + "train=" + trainmu)
    else:
        Y, Yerr = [], []
        for listPath in graphIndex:
            if mesure == "NMI":
                Y.append(1 - np.mean( [dictRes[path][curbname] for path in listPath if curbname in dictRes[path]] ))
                # Yerr.append(np.std(1 - [dictRes[path][curbname] for path in listPath if curbname in dictRes[path]]))
            else:
                Y.append(np.mean( [dictRes[path][curbname] for path in listPath if curbname in dictRes[path]] ))
        plt.plot(XIndex, Y, color=next(cmap), label=curbname)
plt.ylabel(mesure)
plt.legend()
plt.savefig(f"{mesure}ComparaisonOfImprovePLPAndOtherMethods.pdf")
