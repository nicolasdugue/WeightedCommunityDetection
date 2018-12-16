import networkit as nk
import random
from collections import Counter

def __PLP(G, startDict, getNeighbors):
    dicoLabels, nbIter, nodes = startDict(G), 0,  G.nodes()
    acha = []
    change = True
    while change:
        ordre = list(range(len(nodes)))
        random.shuffle(ordre)
        nb_change = 0
        for n in ordre:
            voisins = getNeighbors(n)
            if (len(voisins) > 0):
                labels = [dicoLabels[v] for v in voisins]
                c = Counter(labels)
                old_label_count = c[dicoLabels[n]]
                newlabel, newcount = c.most_common(1)[0]
                if newlabel != dicoLabels[n] and newcount > old_label_count:
                    dicoLabels[n] = newlabel
                    nb_change += 1
        nbIter += 1
        acha.append(nb_change)
        if nb_change == 0 or nbIter > 100:
            change = False
    print(Counter(acha))
    print("Nb Iter", nbIter)
    continousCid = {c: i for i, c in enumerate(sorted(list(set(dicoLabels.values()))))}
    return nk.community.Partition(len(nodes),[continousCid[dicoLabels[n]] for n in sorted(dicoLabels)])

def PLP(G):
    def allToSingletons(G):
        return {n:i for i, n in enumerate(G.nodes())}
    
    return __PLP(G, startDict=allToSingletons,
                    getNeighbors=G.neighbors)

def PLPImproveNico(G, inter, intra):
    def startDict(G):
        dicoLabels, cpt = dict(), 0
        for n in G.nodes():
            if n not in dicoLabels:
                dicoLabels[n] = cpt
                voisins = intra[n]
                for v in voisins:
                    dicoLabels[n] = cpt
                cpt += 1
        return dicoLabels

    def getNeighbors(n):
        voisins = set(G.neighbors(n))
        not_voisins = set(inter[n])
        return voisins.difference(not_voisins)

    return __PLP(G, startDict=startDict,
                    getNeighbors=getNeighbors)
