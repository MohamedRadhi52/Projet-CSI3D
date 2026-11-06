#!/usr/bin/env python3
"""
Base de comparaison des critères de simplification.

Pour chaque modèle et chaque critère : compression en .obja (progressive_mesh.py),
puis décompression progressive. À chaque ligne 's' du fichier, on mesure l'écart
entre le maillage reçu et l'original :
  - erreur globale : distance quadratique moyenne (RMS) dans les deux sens ;
  - erreur sur les zones saillantes : RMS depuis les 20 % de points de l'original
    les plus saillants (ce que l'oeil regarde).
Erreurs en % de la diagonale de la boîte englobante.

Sorties dans resultats/ : debit_distorsion.png, apercu.png, tableau.md et les .obja.
Usage : python3 evaluate.py      (numpy, scipy, matplotlib)
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree

import progressive_mesh as pmesh

MODELES = ['suzanne', 'bunny']
N_POINTS = 8000
BUDGETS = [0.1, 0.2, 0.3, 0.5]          # part du fichier QEM reçue, pour le tableau


class LongueurArete(pmesh.ProgressiveMesh):
    """Ancien critère : la plus courte arête d'abord."""
    def edge_cost(self, a, b):
        return (float(np.linalg.norm(self.V[a] - self.V[b])), a, b, self.V[a])


# fichier, nom, classe, LAMBDA, PROTECT, couleur, style
CRITERES = [
    ('longueur', "Longueur d'arête", LongueurArete, 0, 0, '#898781', '--'),
    ('qem', 'QEM', pmesh.ProgressiveMesh, 0, 0, '#2a78d6', '-'),
    ('seuil', 'QEM + seuil de saillance (article)', pmesh.ProgressiveMesh, 0, 0.2, '#eb6834', '-'),
    ('lambda2', 'QEM + saillance pondérée, λ = 2', pmesh.ProgressiveMesh, 2, 0, '#1baf7a', '-'),
    ('lambda10', 'QEM + saillance pondérée, λ = 10', pmesh.ProgressiveMesh, 10, 0, '#eda100', '-'),
]
PLANCHER = 1e-3                          # erreurs plus petites : considérées comme nulles


# Distances point-surface exactes ---------------------------------------------
def point_triangle(P, T):
    """Distance de points P (..., 3) à des triangles T (..., 3, 3) (Ericson, 2004)."""
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
    """Distance exacte de chaque point de X à la surface (V, F)."""
    T = V[F]
    C = T.mean(1)
    R = np.linalg.norm(T - C[:, None], axis=2).max(1)                   # rayon de chaque triangle
    _, near = cKDTree(C).query(X, k=min(8, len(F)))
    d = np.nanmin(point_triangle(X[:, None], T[near.reshape(len(X), -1)]), axis=1)   # majorant
    level = np.floor(np.log2(np.maximum(R, R.max() * 1e-6) / R.max())).astype(int)
    for lv in np.unique(level):            # tous les triangles qui peuvent faire mieux, par taille
        ids = np.flatnonzero(level == lv)
        found = cKDTree(C[ids]).query_ball_point(X, d + R[ids].max())
        pts = np.repeat(np.arange(len(X)), [len(f) for f in found])
        if len(pts):
            np.minimum.at(d, pts, point_triangle(X[pts], T[ids[np.concatenate(found).astype(int)]]))
    return d


def sample(V, F, n, rng):
    """n points uniformes sur la surface : positions, triangle et coordonnées (u, v)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    f = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    u, v = np.where(u + v > 1, 1 - u, u), np.where(u + v > 1, 1 - v, v)
    return a[f] + u[:, None] * (b[f] - a[f]) + v[:, None] * (c[f] - a[f]), f, u, v


# Mesures -----------------------------------------------------------------------
def evaluer(modele):
    src = f'example/{modele}.obj'
    ref = pmesh.ProgressiveMesh()
    ref.load(src)
    ref.collect_edges()
    S0, V0, F0 = ref.compute_saliency(), ref.V, ref.F
    diag = np.linalg.norm(V0.max(0) - V0.min(0))
    X, f, u, v = sample(V0, F0, N_POINTS, np.random.default_rng(0))
    s = (1 - u - v) * S0[F0[f, 0]] + u * S0[F0[f, 1]] + v * S0[F0[f, 2]]
    saillant = s >= np.quantile(s, 0.8)
    courbes, etats = {}, {}
    for fichier_id, nom, classe, lam, protect, *_ in CRITERES:
        pmesh.LAMBDA, pmesh.PROTECT = lam, protect
        pm = classe()
        pm.prepare(src)
        pm.simplify(0.0)
        dst = f'resultats/{modele}_{fichier_id}.obja'
        with open(dst, 'w') as fichier:
            pm.write_obja(fichier)
        assert pmesh.check(src, dst), dst
        lignes = []
        for i, (octets, V, F) in enumerate(pmesh.decode(dst)):
            d1 = distance(X, V, F)
            d2 = distance(sample(V, F, N_POINTS, np.random.default_rng(i))[0], V0, F0)
            lignes.append((octets / 1024, np.sqrt(np.mean(np.r_[d1, d2] ** 2)) / diag * 100,
                           np.sqrt(np.mean(d1[saillant] ** 2)) / diag * 100))
            etats.setdefault(nom, []).append((octets, V, F))
        courbes[nom] = np.array(lignes)
        print(f'{modele:8s} {nom:36s} {lignes[-1][0]:6.0f} Ko, décompression exacte')
    return courbes, etats, (V0, F0, S0)


def erreur_a(courbe, ko):
    """Erreur au dernier point de contrôle reçu avant ko kilo-octets."""
    return courbe[max(0, np.searchsorted(courbe[:, 0], ko, side='right') - 1)]


# Figures -----------------------------------------------------------------------
def courbes_debit_distorsion(resultats):
    fig, axes = plt.subplots(len(MODELES), 2, figsize=(11, 4.2 * len(MODELES)), dpi=150)
    for ligne, modele in zip(axes, MODELES):
        for col, (ax, titre) in enumerate(zip(ligne, ['erreur globale', 'erreur sur les zones saillantes'])):
            for _, nom, *_, couleur, style in CRITERES:
                c = resultats[modele][0][nom]
                ok = c[:, col + 1] > PLANCHER
                ax.plot(c[ok, 0], c[ok, col + 1], style, color=couleur, lw=1.3, label=nom)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(bottom=PLANCHER)
            ax.set_title(f'{modele.capitalize()} : {titre}', fontsize=11, color='#0b0b0b', loc='left')
            ax.set_xlabel('Octets reçus (Ko)', color='#52514e')
            ax.set_ylabel('Erreur RMS (% de la diagonale)', color='#52514e')
            ax.grid(True, which='both', color='#e1e0d9', lw=0.5)
            ax.tick_params(colors='#52514e', labelsize=9)
            for side in ax.spines.values():
                side.set_color('#c3c2b7')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig('resultats/debit_distorsion.png', facecolor='white')


def dessiner(ax, V, F, couleurs, rot, centre, rayon):
    a, b = np.radians(rot)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    T = ((V - centre) @ (Rx @ Ry).T)[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    lumiere = np.array([-0.35, 0.45, 1.0]) / np.linalg.norm([-0.35, 0.45, 1.0])
    ombre = np.where(n[:, 2] > 0, 0.35 + 0.65 * np.clip(n @ lumiere, 0, 1), 0.2)
    ordre = np.argsort(T[:, :, 2].mean(1))                              # peintre : du fond vers l'avant
    ax.add_collection(PolyCollection(T[ordre][:, :, :2], facecolors=couleurs[ordre] * ombre[ordre, None],
                                     edgecolors=(0.1, 0.1, 0.12, 0.45), linewidths=0.2))
    ax.set_xlim(-rayon, rayon)
    ax.set_ylim(-rayon, rayon)
    ax.set_aspect('equal')
    ax.axis('off')


def apercu(resultats, part=0.15):
    vues = {'suzanne': (-28, 12), 'bunny': (10, 8)}
    montres = [c[1:] for c in CRITERES if c[0] != 'longueur']
    fig, axes = plt.subplots(len(MODELES), len(montres) + 1, figsize=(3.2 * (len(montres) + 1), 3.6 * len(MODELES)), dpi=130)
    for ligne, modele in zip(axes, MODELES):
        courbes, etats, (V0, F0, S0) = resultats[modele]
        used = np.unique(F0)
        centre = (V0[used].max(0) + V0[used].min(0)) / 2
        rayon = np.linalg.norm(V0[used] - centre, axis=1).max() * 0.92
        dessiner(ligne[0], V0, F0, plt.get_cmap('inferno')(0.1 + 0.9 * S0[F0].mean(1))[:, :3], vues[modele], centre, rayon)
        ligne[0].set_title(f'{modele.capitalize()} : saillance', fontsize=10)
        budget = part * courbes['QEM'][-1, 0] * 1024
        for ax, (nom, *_) in zip(ligne[1:], montres):
            octets, V, F = [e for e in etats[nom] if e[0] <= budget][-1]
            dessiner(ax, V, F, np.tile([0.86, 0.80, 0.70], (len(F), 1)), vues[modele], centre, rayon)
            ax.set_title(f'{nom}\n{len(F)} faces, {octets / 1024:.0f} Ko reçus', fontsize=9)
    fig.suptitle(f'Maillages reçus après {int(part * 100)} % du fichier', fontsize=11)
    fig.tight_layout()
    fig.savefig('resultats/apercu.png', facecolor='white')


def tableau(resultats):
    lignes = ['Erreur RMS en % de la diagonale : globale / zones saillantes, '
              'selon la part du fichier QEM déjà reçue.', '']
    for modele in MODELES:
        courbes = resultats[modele][0]
        total = courbes['QEM'][-1, 0]
        lignes += [f'**{modele}** (fichier QEM : {total:.0f} Ko)', '',
                   '| Critère | ' + ' | '.join(f'{int(b * 100)} %' for b in BUDGETS) + ' |',
                   '|---|' + '---|' * len(BUDGETS)]
        for _, nom, *_ in CRITERES:
            vals = [erreur_a(courbes[nom], b * total) for b in BUDGETS]
            lignes.append(f'| {nom} | ' + ' | '.join(f'{v[1]:.3f} / {v[2]:.3f}' for v in vals) + ' |')
        lignes.append('')
    with open('resultats/tableau.md', 'w') as fichier:
        fichier.write('\n'.join(lignes))
    print('\n'.join(lignes))


if __name__ == '__main__':
    os.makedirs('resultats', exist_ok=True)
    resultats = {modele: evaluer(modele) for modele in MODELES}
    courbes_debit_distorsion(resultats)
    apercu(resultats)
    tableau(resultats)
