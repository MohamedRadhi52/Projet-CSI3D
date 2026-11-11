#!/usr/bin/env python3

import heapq
import sys

import numpy as np

import obja


def voisinage(modele):
    """Retourne les voisins de chaque sommet et les sommets du bord."""
    voisins = {v: set() for v in range(len(modele.vertices))}
    nb_faces = {}
    for face in modele.faces:
        coins = (face.a, face.b, face.c)
        if not face.visible or len(set(coins)) < 3:
            continue
        for k in range(3):
            a, b = coins[k], coins[(k + 1) % 3]
            voisins[a].add(b)
            voisins[b].add(a)
            arete = (min(a, b), max(a, b))
            nb_faces[arete] = nb_faces.get(arete, 0) + 1
    bord = {v for arete, n in nb_faces.items() if n == 1 for v in arete}
    return voisins, bord


def courbure_moyenne(modele):
    """Calcule la courbure moyenne (cotangentes) et l'aire de chaque sommet."""
    sommets = modele.vertices
    num_v = len(sommets)
    laplacien = np.zeros((num_v, 3))
    normales = np.zeros((num_v, 3))
    aires = np.zeros(num_v)
    for face in modele.faces:
        coins = (face.a, face.b, face.c)
        if not face.visible or len(set(coins)) < 3:
            continue
        for k in range(3):
            i, j, l = coins[k], coins[(k + 1) % 3], coins[(k + 2) % 3]
            u = sommets[j] - sommets[i]
            w = sommets[l] - sommets[i]
            produit = np.cross(u, w)
            norme = float(np.linalg.norm(produit))
            cot = float(np.dot(u, w)) / max(norme, 1e-12)
            laplacien[j] += cot / 2.0 * (sommets[l] - sommets[j])
            laplacien[l] += cot / 2.0 * (sommets[j] - sommets[l])
            aires[i] += norme / 6.0
            normales[i] += produit
    normales /= np.maximum(np.linalg.norm(normales, axis=1, keepdims=True), 1e-12)
    courbure = -(laplacien * normales).sum(axis=1) / (2.0 * np.maximum(aires, 1e-12))
    return courbure, aires


def voisinage_geodesique(sommets, voisins, depart, rayon):
    """Retourne la distance géodésique (Dijkstra) entre depart et les sommets à moins de rayon."""
    distances = {depart: 0.0}
    tas = [(0.0, depart)]
    while tas:
        distance, sommet = heapq.heappop(tas)
        if distance > distances[sommet]:
            continue
        for voisin in voisins[sommet]:
            nouvelle = distance + float(np.linalg.norm(sommets[voisin] - sommets[sommet]))
            if nouvelle <= rayon and nouvelle < distances.get(voisin, float("inf")):
                distances[voisin] = nouvelle
                heapq.heappush(tas, (nouvelle, voisin))
    return distances


def entropie(classes, poids):
    """Calcule l'entropie de l'histogramme des classes de courbure, pondéré par les aires."""
    histogramme = np.bincount(classes, weights=poids, minlength=256)
    total = float(histogramme.sum())
    if total <= 0.0:
        return 0.0
    p = histogramme[histogramme > 0] / total
    return float(-(p * np.log2(p)).sum())


def calculer_saillance(modele):
    """Calcule la saillance multi-échelle LCE (Limper 2016) pour chaque sommet, entre 0 et 1."""
    sommets = modele.vertices
    num_v = len(sommets)
    saillance = np.zeros(num_v, dtype=np.float64)
    voisins, bord = voisinage(modele)

    # 1. Courbure moyenne discrète, répartie en 256 classes
    courbure, aires = courbure_moyenne(modele)
    rayons = 0.08 * np.sqrt(aires.sum()) / 2.0 ** np.arange(5)
    utilises = [v for v in range(num_v) if voisins[v]]
    bas, haut = np.percentile(courbure[utilises], [1, 99])
    classes = np.clip(((courbure - bas) / max(haut - bas, 1e-12) * 256).astype(int), 0, 255)
    aires[list(bord)] = 0.0

    # 2-3. Entropie locale de courbure sur 5 voisinages géodésiques
    for v in utilises:
        distances = voisinage_geodesique(sommets, voisins, v, rayons[0])
        autour = np.array(list(distances.keys()))
        ecarts = np.array(list(distances.values()))
        for rayon in rayons:
            proches = autour[ecarts <= rayon]
            saillance[v] += entropie(classes[proches], aires[proches])

    # 4. Lissage avec les voisins puis normalisation finale dans [0, 1]
    lisse = np.array([
        (saillance[v] + sum(saillance[u] for u in voisins[v])) / (1 + len(voisins[v]))
        for v in range(num_v)
    ])
    mini, maxi = lisse[utilises].min(), lisse[utilises].max()
    return np.clip((lisse - mini) / max(maxi - mini, 1e-12), 0.0, 1.0)


if __name__ == "__main__":
    modele = obja.parse_file(sys.argv[1] if len(sys.argv) > 1 else "example/bunny.obj")
    s = calculer_saillance(modele)
    print(f"{len(s)} sommets : saillance entre {s.min():.2f} et {s.max():.2f}, moyenne {s.mean():.2f}")
