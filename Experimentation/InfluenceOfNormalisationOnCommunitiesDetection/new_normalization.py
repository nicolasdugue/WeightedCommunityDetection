from numpy import log2
import networkit as nk

def pmi(depG):
    G = nk.Graph(depG, weighted=True)
    edges_list = depG.edges()
    dico = dict()
    tot = depG.totalEdgeWeight()
    for n1, n2 in edges_list:
        w1 = depG.weightedDegree(n1)
        w2 = depG.weightedDegree(n2)
        w = depG.weight(n1, n2)
        try:
            pmi = log2((w * tot)/(w1 * w2))
        except ZeroDivisionError as e:
            pmi = 0
        dico[(n1, n2)] = pmi
    for n1, n2 in dico:
        if dico[(n1, n2)] != 0:
            G.setWeight(n1, n2, dico[(n1, n2)])
        else:
            G.removeEdge(n1, n2)
    return G


def ppmi(depG):
    G = nk.Graph(depG, weighted=True)
    edges_list = depG.edges()
    dico = dict()
    tot = depG.totalEdgeWeight()
    for n1, n2 in edges_list:
        w1 = depG.weightedDegree(n1)
        w2 = depG.weightedDegree(n2)
        w = depG.weight(n1, n2)
        try:
            pmi = log2((w * tot)/(w1 * w2))
        except ZeroDivisionError as e:
            pmi = 0
        dico[(n1, n2)] = pmi
    for n1, n2 in dico:
        if dico[(n1, n2)] != 0:
            G.setWeight(n1, n2, max(0, dico[(n1, n2)]))
        else:
            G.removeEdge(n1, n2)
    return G


def fake_pmi(depG):
    G = nk.Graph(depG, weighted=True)
    edges_list = depG.edges()
    dico = dict()
    tot = depG.totalEdgeWeight()
    for n1, n2 in edges_list:
        w1 = depG.weightedDegree(n1)
        w2 = depG.weightedDegree(n2)
        w = depG.weight(n1, n2)
        try:
            pmi = log2(w / (w1 * w2))
        except ZeroDivisionError as e:
            # print(w1, w2, w)
            pmi = 0
        dico[(n1, n2)] = pmi
    for n1, n2 in dico:
        if dico[(n1, n2)] != 0:
            G.setWeight(n1, n2, max(0, dico[(n1, n2)]))
        else:
            G.removeEdge(n1, n2)
    return G


def standard(depG):
    G = nk.Graph(depG, weighted=True)
    edges_list = depG.edges()
    dico = dict()
    tot = depG.totalEdgeWeight()
    for n1, n2 in edges_list:
        w1 = depG.weightedDegree(n1)
        w2 = depG.weightedDegree(n2)
        w = depG.weight(n1, n2)
        try:
            res = w / (w1 * w2)
        except ZeroDivisionError as e:
            # print(w1, w2, w)
            res = 0
        dico[(n1, n2)] = res
    for n1, n2 in dico:
        if dico[(n1, n2)] != 0:
            G.setWeight(n1, n2, dico[(n1, n2)])
        else:
            G.removeEdge(n1, n2)
    return G


def glove(depG, xmax=100, alpha=0.75):
    G = nk.Graph(depG, weighted=True)
    edges_list = depG.edges()
    dico = dict()
    tot = depG.totalEdgeWeight()
    res = -1
    for n1, n2 in edges_list:
        w = depG.weight(n1, n2)
        if (w > xmax):
            res = 1
        else:
            res = (w / xmax) ** alpha
        dico[(n1, n2)] = res
    for n1, n2 in dico:
        if dico[(n1, n2)] != 0:
            G.setWeight(n1, n2, dico[(n1, n2)])
        else:
            G.removeEdge(n1, n2)
    return G


def iterative_rev_degree_order(depG):
    G = nk.Graph(depG, weighted=True)
    edges = sorted(G.edges(), key=lambda x: G.weightedDegree(x[0]) + G.weightedDegree(x[1]), reverse=True)
    for u, v in edges:
        try:
            w = G.weight(u, v) / (G.weightedDegree(u) * G.weightedDegree(v))
        except ZeroDivisionError as e:
            continue
        if w == 0:
            G.removeEdge(u, v)
        else:
            G.setWeight(u, v, w=w)
    return G
