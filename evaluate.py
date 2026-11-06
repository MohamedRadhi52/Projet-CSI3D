#!/usr/bin/env python3
"""
Compare les critères de simplification en affichant leurs erreurs.

Pour chaque modèle et chaque critère : on compresse en .obja, puis on regarde
le maillage reçu quand 10, 20, 30 et 50 % du fichier sont arrivés, et on mesure
son écart avec l'original, en % de la taille du modèle :
  - erreur globale : sur toute la surface ;
  - erreur saillante : sur les 20 % de la surface les plus saillants.
Plus c'est petit, mieux c'est.

Usage : python3 evaluate.py      (numpy et scipy, moins d'une minute)
"""
import os
import tempfile

import numpy as np
from scipy.spatial import cKDTree

import progressive_mesh as P

MODELES = ['suzanne', 'bunny']
PARTS = [0.1, 0.2, 0.3, 0.5]          # part du fichier reçue au moment de la mesure


class LongueurArete(P.ProgressiveMesh):
    """Ancien critère : la plus courte arête d'abord."""
    def edge_cost(self, a, b):
        return (float(np.linalg.norm(self.V[a] - self.V[b])), a, b, self.V[a])


CRITERES = [  # nom, classe, LAMBDA, PROTECT
    ("Longueur d'arête", LongueurArete, 0, 0),
    ('QEM', P.ProgressiveMesh, 0, 0),
    ('Saillance, seuil (article)', P.ProgressiveMesh, 0, 0.2),
    ('Saillance pondérée, λ = 2', P.ProgressiveMesh, 2, 0),
    ('Saillance pondérée, λ = 10', P.ProgressiveMesh, 10, 0),
]


def evaluer(modele, dossier):
    """{critère: [(erreur globale, erreur saillante) pour chaque part du fichier]}"""
    src = f'example/{modele}.obj'
    ref = P.ProgressiveMesh()
    ref.load(src)
    ref.collect_edges()
    V0, F0, S0 = ref.V, ref.F, ref.compute_saliency()
    taille = np.linalg.norm(V0.max(0) - V0.min(0))       # diagonale de la boîte du modèle
    X, sx = points_sur(V0, F0, S0)                        # points tirés sur l'original
    saillant = sx >= np.quantile(sx, 0.8)                 # les 20 % les plus saillants

    fichiers = {}                                         # 1. compression avec chaque critère
    for i, (nom, classe, lam, protect) in enumerate(CRITERES):
        P.LAMBDA, P.PROTECT = lam, protect
        pm = classe()
        pm.prepare(src)
        pm.simplify(0.0)
        fichiers[nom] = f'{dossier}/{modele}_{i}.obja'
        with open(fichiers[nom], 'w') as f:
            pm.write_obja(f)

    total = os.path.getsize(fichiers['QEM'])              # 2. mesures, à octets égaux
    resultats = {}
    for nom, chemin in fichiers.items():
        resultats[nom] = []
        for part in PARTS:
            V, F = recu(chemin, part * total)
            d1 = distance(X, V, F)                        # original -> reçu
            d2 = distance(points_sur(V, F)[0], V0, F0)    # reçu -> original
            resultats[nom].append((rms(np.r_[d1, d2]) / taille * 100, rms(d1[saillant]) / taille * 100))
    return resultats


def recu(chemin, octets):
    """Maillage reçu quand 'octets' octets sont arrivés : état à la dernière ligne 's' avant."""
    etat = None
    for nb, V, F in P.decode(chemin):
        if nb > octets:
            break
        etat = V, F
    return etat


def afficher(modele, resultats):
    entete = f'{"critère":30s}' + ''.join(f'{int(p * 100):>8d} %' for p in PARTS)
    for k, titre in enumerate(['Erreur globale', 'Erreur sur les zones saillantes']):
        print(f'\n{modele} - {titre} (% de la taille du modèle), selon la part du fichier reçue')
        print(entete)
        for nom, erreurs in resultats.items():
            print(f'{nom:30s}' + ''.join(f'{e[k]:10.3f}' for e in erreurs))


# Outils de mesure ----------------------------------------------------------------
def rms(d):
    """Moyenne quadratique des distances."""
    return np.sqrt(np.mean(d ** 2))


def points_sur(V, F, S=None, n=8000):
    """n points tirés au hasard sur la surface (et leur saillance si S est donnée)."""
    rng = np.random.default_rng(0)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    aire = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    f = rng.choice(len(F), n, p=aire / aire.sum())        # un triangle, au prorata de l'aire
    u, v = rng.random(n), rng.random(n)
    u, v = np.where(u + v > 1, 1 - u, u), np.where(u + v > 1, 1 - v, v)
    X = a[f] + u[:, None] * (b[f] - a[f]) + v[:, None] * (c[f] - a[f])
    s = None if S is None else (1 - u - v) * S[F[f, 0]] + u * S[F[f, 1]] + v * S[F[f, 2]]
    return X, s


def point_triangle(P, T):
    """Distance de points P (..., 3) à des triangles T (..., 3, 3) : point le plus proche
    sur le triangle, à l'intérieur, sur une arête ou sur un sommet (Ericson, 2004)."""
    a, b, c = T[..., 0, :], T[..., 1, :], T[..., 2, :]
    ab, ac = b - a, c - a
    d1, d2 = (ab * (P - a)).sum(-1), (ac * (P - a)).sum(-1)
    d3, d4 = (ab * (P - b)).sum(-1), (ac * (P - b)).sum(-1)
    d5, d6 = (ab * (P - c)).sum(-1), (ac * (P - c)).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide='ignore', invalid='ignore'):
        den = va + vb + vc
        Q = a + ab * (vb / den)[..., None] + ac * (vc / den)[..., None]     # intérieur
        for zone, q in [((va <= 0) & (d4 >= d3) & (d5 >= d6), b + (c - b) * ((d4 - d3) / (d4 - d3 + d5 - d6))[..., None]),
                        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * (d2 / (d2 - d6))[..., None]),
                        ((d6 >= 0) & (d5 <= d6), c),
                        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * (d1 / (d1 - d3))[..., None]),
                        ((d3 >= 0) & (d4 <= d3), b),
                        ((d1 <= 0) & (d2 <= 0), a)]:                       # arêtes puis sommets
            Q = np.where(zone[..., None], q, Q)
    return np.linalg.norm(P - Q, axis=-1)


def distance(X, V, F):
    """Distance exacte de chaque point de X à la surface (V, F) : on ne teste que les
    triangles assez proches pour pouvoir être le plus proche."""
    T = V[F]
    C = T.mean(1)
    R = np.linalg.norm(T - C[:, None], axis=2).max(1)                   # rayon de chaque triangle
    _, near = cKDTree(C).query(X, k=min(8, len(F)))
    d = np.nanmin(point_triangle(X[:, None], T[near.reshape(len(X), -1)]), axis=1)   # majorant
    niveau = np.floor(np.log2(np.maximum(R, R.max() * 1e-6) / R.max())).astype(int)
    for nv in np.unique(niveau):           # triangles regroupés par taille
        ids = np.flatnonzero(niveau == nv)
        trouves = cKDTree(C[ids]).query_ball_point(X, d + R[ids].max())
        pts = np.repeat(np.arange(len(X)), [len(t) for t in trouves])
        if len(pts):
            np.fmin.at(d, pts, point_triangle(X[pts], T[ids[np.concatenate(trouves).astype(int)]]))
    return d


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as dossier:   # fichiers .obja effacés à la fin
        for modele in MODELES:
            afficher(modele, evaluer(modele, dossier))
