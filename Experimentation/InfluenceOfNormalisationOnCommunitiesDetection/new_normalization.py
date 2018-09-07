from numpy import log2


def pmi(G):
    edges_list = G.edges()
    dico = dict()
    tot = G.totalEdgeWeight()
    for n1, n2 in edges_list:
        w1 = G.weightedDegree(n1)
        w2 = G.weightedDegree(n2)
        w = G.weight(n1, n2)
        pmi = log2(
            (w / tot)
            /
            ((w1 * w2) / (tot ** 2))
            )
        dico[(n1, n2)] = pmi
    for n1, n2 in dico:
        if dico[(n1, n2)] == 0:
            G.removeEdge(n1, n2)
        else:
            G.setWeight(n1, n2, dico[(n1, n2)])
    return G


def ppmi(G):
    edges_list = G.edges()
    dico = dict()
    tot = G.totalEdgeWeight()
    for n1, n2 in edges_list:
        w1 = G.weightedDegree(n1)
        w2 = G.weightedDegree(n2)
        w = G.weight(n1, n2)
        pmi = log2(
            (w / tot)
            /
            ((w1 * w2) / (tot ** 2))
            )
        dico[(n1, n2)] = pmi
    for n1, n2 in dico:
        if dico[(n1, n2)] == 0:
            G.removeEdge(n1, n2)
        else:
            G.setWeight(n1, n2, max(0, dico[(n1, n2)]))
    return G


def fake_pmi(G):
    edges_list = G.edges()
    dico = dict()
    tot = G.totalEdgeWeight()
    for n1, n2 in edges_list:
        w1 = G.weightedDegree(n1)
        w2 = G.weightedDegree(n2)
        w = G.weight(n1, n2)
        try:
            pmi = log2(w / (w1 * w2))
        except ZeroDivisionError as e:
            print(w1, w2, w)
            pmi = 0
        dico[(n1, n2)] = pmi
    for n1, n2 in dico:
        if dico[(n1, n2)] == 0:
            G.removeEdge(n1, n2)
        else:
            G.setWeight(n1, n2, max(0, dico[(n1, n2)]))
    return G


def standard(G):
    edges_list = G.edges()
    dico = dict()
    tot = G.totalEdgeWeight()
    for n1, n2 in edges_list:
        w1 = G.weightedDegree(n1)
        w2 = G.weightedDegree(n2)
        w = G.weight(n1, n2)
        res = w / (w1 * w2)
        dico[(n1, n2)] = res
    for n1, n2 in dico:
        if dico[(n1, n2)] == 0:
            G.removeEdge(n1, n2)
        else:
            G.setWeight(n1, n2, dico[(n1, n2)])
    return G


def glove(G, xmax=100, alpha=0.75):
    edges_list = G.edges()
    dico = dict()
    tot = G.totalEdgeWeight()
    res = -1
    for n1, n2 in edges_list:
        w = G.weight(n1, n2)
        if (w > xmax):
            res = 1
        else:
            res = (w / xmax) ** alpha
        dico[(n1, n2)] = res
    for n1, n2 in dico:
        if dico[(n1, n2)] == 0:
            G.removeEdge(n1, n2)
        else:
            G.setWeight(n1, n2, dico[(n1, n2)])
    return G


def iterative_rev_degree_order(G):
    edges = sorted(G.edges(), key=lambda x: G.weightedDegree(x[0]) + G.weightedDegree(x[1]), reverse=True)
    for u, v in edges:
        w = G.weight(u, v) / (G.weightedDegree(u) * G.weightedDegree(v))
        if w == 0:
            G.removeEdge(u, v)
        else:
            G.setWeight(u, v, w=w)
    return G
