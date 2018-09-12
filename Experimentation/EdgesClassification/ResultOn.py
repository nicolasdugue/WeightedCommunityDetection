import argparse
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import networkit as nk
import numpy as np
import xgboost as xgb
# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("path", help="Directory with network and community", type=str)
args = argparser.parse_args()
path = args.path
# %%
# path = "/home/vconnes/WeightedCommunityDetection/lfr_5000/mk100/k20/muw0.4/4/"
print("__LOADINGS__")
# loading of graph
G = nk.graphio.readGraph(os.path.join(path, "network.dat"), weighted=True, fileformat=nk.Format.EdgeListTabOne)
removed = []
for u, v in G.edges():
    if G.weight(u, v) == 0:
        removed.append((u, v))
res = dict(numberOfnodes=G.numberOfNodes(), numberOfEdges=G.numberOfEdges(),
           percentOfNulWeight=len([1 for u, v in G.edges() if G.weight(u, v) == 0])/G.numberOfEdges())
for (u, v) in removed:
    G.removeEdge(u, v)
nk.overview(G)
tot = G.totalEdgeWeight()
print(tot)
# loading of communities
evalname = "Groundtruth"
print(f"__{evalname}__")
gt_partition = nk.community.readCommunities(os.path.join(path, "community.dat"), format="edgelist-t1")
nk.community.inspectCommunities(gt_partition, G)
res["numberOfCom" + evalname] = gt_partition.numberOfSubsets()
print(f"{gt_partition.numberOfSubsets()} community detected")

edges = G.edges()
deg_min = []
deg_max = []
clust_min = []
clust_max = []
weight = []
inside = []
cc = nk.centrality.LocalClusteringCoefficient(G).run().scores()

for (u, v) in edges:
    degU, degV = G.weightedDegree(u), G.weightedDegree(v)
    clustU, clustV = cc[u], cc[v]

    deg_min.append(min(degU, degV))
    deg_max.append(max(degU, degV))
    clust_min.append(min(clustU, clustV))
    clust_max.append(max(clustU, clustV))
    weight.append(G.weight(u, v))

    if gt_partition.subsetOf(u) == gt_partition.subsetOf(v):
        inside.append(1)
    else:
        inside.append(0)

target = ["outside", "inside"]
features = ["deg_min", "deg_max", "clust_min", "clust_max", "weight"]
X = np.array([deg_min, deg_max, clust_min, clust_max, weight])
Y = inside
X = X.transpose()
samples, features = X.shape
print(f"{features} features on {samples} samples")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,
                                                    random_state=0)
print(f"Trainning set:{len(X_train)} samples")
print(f"Testing set:{len(X_test)} samples")

gbm = xgb.XGBClassifier(max_depth=8, n_estimators=300, learning_rate=0.05).fit(X_train, Y_train)
predictions = gbm.predict(X_test)

# %%
# xgb.plot_tree(gbm, num_trees=2)
# fig = plt.gcf()
# fig.set_size_inches(10, 10)
# fig.savefig('tree.png')
# %%
print(metrics.classification_report(Y_test, predictions))
mat = metrics.confusion_matrix(Y_test, predictions)
print("Confusion matrix:")
print(mat)
prec, rec, fmeasure, support = metrics.precision_recall_fscore_support(Y_test, predictions)
print("Importance of features:")
weights = gbm.feature_importances_
print(weights)
# xgb.plot_importance(gbm)
# %%
# pred_score = gbm.predict_proba(X_test)
# pred_score = [pred_score[i][val] for i, val in enumerate(Y_test)]
# precisiondata = dict()
# recalldata = dict()
# average_precision_data = dict()
#
# for i in range(len(target)):
#     indexs = [j for j, x in enumerate(Y_test) if x == i]
#     Y_class = [Y_test[j] for j in indexs]
#     pred_class = [pred_score[j] for j in indexs]
#     precisiondata[i], recalldata[i], _ = metrics.precision_recall_curve(Y_class, pred_class)
#     average_precision_data[i] = metrics.average_precision_score(Y_class, pred_class)
#
# # A "micro-average": quantifying score on all classes jointly
# precisiondata["micro"], recalldata["micro"], _ = metrics.precision_recall_curve(Y_test, pred_score)
# average_precision_data["micro"] = metrics.average_precision_score(Y_test, pred_score, average="micro")
#
# data = (precisiondata, recalldata, average_precision_data)

res = dict(target=target, precision=prec, recall=rec, f1=fmeasure, support=support, confmat=mat,
           features=features, weights=weights)
# %%
# from itertools import cycle
# import matplotlib.pyplot as plt
# # setup plot details
# colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
#
# plt.figure(figsize=(7, 8))
# f_scores = np.linspace(0.2, 0.8, num=4)
# lines = []
# labels = []
# for f_score in f_scores:
#     x = np.linspace(0.01, 1)
#     y = f_score * x / (2 * x - f_score)
#     l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
#     plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
#
# lines.append(l)
# labels.append('iso-f1 curves')
# l, = plt.plot(recalldata["micro"], precisiondata["micro"], color='gold', lw=2)
# lines.append(l)
# labels.append('micro-average Precision-recall (area = {0:0.2f})'
#               ''.format(average_precision_data["micro"]))
#
# for i, color in zip(range(len(target)), colors):
#     l, = plt.plot(recalldata[i], precisiondata[i], color=color, lw=2)
#     lines.append(l)
#     labels.append('Precision-recall for class {0} (area = {1:0.2f})'
#                   ''.format(target[i], average_precision_data[i]))
#
# fig = plt.gcf()
# fig.subplots_adjust(bottom=0.25)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Extension of Precision-Recall curve to multi-class')
# plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
#
#
# plt.savefig(f"precrec.pdf")
# %%
with open(os.path.join(path, "xp2.pickle"), "wb") as file:
    pickle.dump(res, file)
