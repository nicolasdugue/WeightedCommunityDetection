import argparse
import os
import pickle
sys.path.append("../Toolbox")
from sklearn.model_selection import train_test_split
from sklearn import metrics
import networkit as nk
import numpy as np
import xgboost as xgb
from Utils import loadings, partitionRes, statNodes
# path = "/home/vconnes/WeightedCommunityDetection/lfr_5000/mk100/k20/muw0.4/4/"
path = "/home/vconnes/WeightedCommunityDetection/lfr_5000/mk100/k20/muw0.4/5/"
# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("path", help="Directory with network and community", type=str)
argparser.add_argument("--addAssort", help="If true assortativity features are used, default=True", action="store_true", default=False)
args = argparser.parse_args()
path = args.path
addAssort = args.addAssort
# %%
G, gt_partition, _ = loadings(path)
tot = G.totalEdgeWeight()
# %%
edges = G.edges()
X, Y, target, features = statNodes(G, gt_partition, edges, addAssort=addAssort)
res = {"target"=target, "features"=features}
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,
                                                    random_state=0)
print(f"Trainning set:{len(X_train)} samples")
print(f"Testing set:{len(X_test)} samples")

# %%
 if addAssort:
        gbm = xgb.XGBClassifier(max_depth=8, n_estimators=300, learning_rate=0.05).fit(X_train, Y_train)
    else:
        gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, Y_train)

predictions = gbm.predict(X_test)
res = statClassifier(gbm, Y, predictions)

# %%
# xgb.plot_importance(gbm)
# xgb.plot_tree(gbm, num_trees=2)
# fig = plt.gcf()
# fig.set_size_inches(10, 10)
# fig.show
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
if addAssort:
    with open(os.path.join(path, "xp2_7.pickle"), "wb") as file:
        pickle.dump(res, file)
else:
    with open(os.path.join(path, "xp2.pickle"), "wb") as file:
        pickle.dump(res, file)
