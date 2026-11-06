#!/usr/bin/env python3
"""Vérifie chaque étape de progressive_mesh.py.
Usage : python3 test_etapes.py [example/bunny.obj]"""
import sys

import numpy as np

import progressive_mesh as P

src = sys.argv[1] if len(sys.argv) > 1 else 'example/bunny.obj'
dst = src.replace('.obj', '_pm.obja')


def ok(etape, condition, detail):
    print(f'{"OK   " if condition else "ÉCHEC"} {etape:28s} {detail}')


def euler(pm):
    """V - E + F du maillage courant : ne change pas si la topologie est conservée."""
    faces = np.unique(np.sort(pm.F[list(set().union(*pm.vf))], axis=1), axis=0)  # face en double : 1 fois
    E = np.unique(faces[:, [0, 1, 1, 2, 0, 2]].reshape(-1, 2), axis=0)
    return len(np.unique(faces)) - len(E) + len(faces)


pm = P.ProgressiveMesh()

# 1. Chargement : chaque face est rangée chez ses 3 sommets
pm.load(src)
ok('1. chargement', all(i in pm.vf[v] for i, f in enumerate(pm.F) for v in f),
   f'{len(pm.V)} sommets, {len(pm.F)} faces, voisinage cohérent')

# 2. Arêtes : pas de doublon, et la formule d'Euler V - E + F
E = pm.collect_edges()
ok('2. arêtes', len(E) == len(set(E)),
   f'{len(E)} arêtes dont {len(pm.border)} de bord, V - E + F = {euler(pm)}')

# 3. Quadriques : un sommet est sur ses propres plans, son erreur vaut 0
pm.compute_quadrics()
err = max(abs(np.append(v, 1) @ q @ np.append(v, 1)) for v, q in zip(pm.V, pm.Q))
ok('3. coût QEM', err < 1e-9, f'erreur d\'un sommet à sa propre position : {err:.0e}')

# 3 bis. Saillance : entre 0 et 1
S = pm.compute_saliency()
ok('3 bis. saillance', S.min() >= 0 and S.max() == 1,
   f'S entre {S.min():.2f} et {S.max():.2f}, moyenne {S.mean():.2f}')

# 4. Choix : les arêtes sortent par coût croissant
pm = P.ProgressiveMesh()               # un objet neuf à chaque fois : load() ajoute au modèle existant
pm.prepare(src)
couts = [pm.pop_best_edge()[3] for _ in range(20)]
ok('4. choix de l\'arête', couts == sorted(couts), '20 arêtes valides sorties par coût croissant')

# 5-7. Une contraction : 1 ou 2 faces en moins, t n'a plus de face, 1 vsplit enregistré
pm = P.ProgressiveMesh()
pm.prepare(src)
avant = len(set().union(*pm.vf))
s, t, p, _ = pm.pop_best_edge()
communes = len(pm.vf[s] & pm.vf[t])
pm.collapse_edge(s, t, p)
retirees = avant - len(set().union(*pm.vf))
ok('5-7. contraction', retirees == communes and not pm.vf[t] and len(pm.history) == 1,
   f'sommet {t} fusionné dans {s} : {retirees} faces en moins, 1 vsplit enregistré')

# 8-9. Simplification à 10 % : même topologie qu'au départ
pm = P.ProgressiveMesh()
pm.prepare(src)
chi = euler(pm)
pm.simplify(0.1)
reste = len(np.unique(pm.F[list(set().union(*pm.vf))]))
ok('8-9. simplification', euler(pm) == chi,
   f'{reste} sommets restants, V - E + F reste égal à {euler(pm)}')

# 10-11. Écriture puis décompression : on retrouve exactement l'original
for lam in (0, 2):
    P.LAMBDA = lam
    pm = P.ProgressiveMesh()
    pm.prepare(src)
    pm.simplify(0.0)
    with open(dst, 'w') as f:
        pm.write_obja(f)
    etats = list(P.decode(dst))
    croissant = all(len(a[2]) <= len(b[2]) for a, b in zip(etats, etats[1:]))
    ok(f'10-11. OBJA (λ = {lam})', P.check(src, dst) and croissant,
       f'base de {len(etats[0][2])} faces -> {len(etats[-1][2])} faces en {len(pm.history)} vsplit, '
       f'{etats[-1][0] // 1024} Ko, décompression exacte')
print(f'\nÀ regarder dans le viewer : ./server.py puis http://localhost:8000/?{dst}')
