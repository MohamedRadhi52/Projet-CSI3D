#!/usr/bin/env python3
"""
Progressive Mesh - étapes 1 à 4.

  1. load()           chargement + voisinage (sommet -> faces)
  2. collect_edges()  arêtes uniques + arêtes de bord
  3. edge_cost()      coût QEM de Garland & Heckbert (remplace edge_length)
  4. pop_best_edge()  tas de priorité + test de validité (remplace shortest_edge)

Étape 5 (à venir) : collapse_edge(s, t, p) supprime les faces partagées par s
et t, remplace t par s dans les autres faces de t (et met self.vf à jour),
place s en p, puis fait self.Q[s] += self.Q[t] et self.update(s).
"""
import heapq
import sys
from itertools import combinations

import numpy as np

import obja

HALF_EDGE = True     # True : s garde sa position (vsplit sans 'ev'), False : position optimale
BORDER_WEIGHT = 1.0  # plans de bord (Garland & Heckbert §6) ; 0 = les trous s'agrandissent


class ProgressiveMesh(obja.Model):

    def prepare(self, path):
        self.load(path)                       # 1
        edges = self.collect_edges()          # 2
        self.compute_quadrics()               # 3
        self.heap, self.candidates = [], {}   # 4 : arête -> (coût, s, t, p) à jour
        for a, b in edges:
            self.push(a, b)

    # 1. Chargement ---------------------------------------------------------
    def load(self, path):
        self.parse_file(path)                                     # parseur d'obja.py
        self.V = np.array(self.vertices)                          # (n, 3) positions
        self.F = np.array([[f.a, f.b, f.c] for f in self.faces])  # (m, 3) triangles
        self.vf = [set() for _ in self.V]                         # sommet -> faces
        for i, face in enumerate(self.F):
            for v in face:
                self.vf[v].add(i)

    def ring(self, v):
        """Sommets voisins de v."""
        return set(self.F[list(self.vf[v])].ravel()) - {v}

    def on_border(self, v):
        return any(len(self.vf[v] & self.vf[x]) == 1 for x in self.ring(v))

    # 2. Arêtes -------------------------------------------------------------
    def collect_edges(self):
        """Arêtes (i < j) sans doublon. Vue par une seule face = arête de bord."""
        E = np.sort(self.F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        E, count = np.unique(E, axis=0, return_counts=True)
        self.border = [tuple(e) for e in E[count == 1]]
        return [tuple(e) for e in E]

    # 3. Coût ---------------------------------------------------------------
    def compute_quadrics(self):
        """Q[v] = somme des p.pT sur les plans p = (a, b, c, d) des faces autour de v."""
        p0, p1, p2 = (self.V[self.F[:, k]] for k in range(3))
        n = np.cross(p1 - p0, p2 - p0)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        planes = np.c_[n, -(n * p0).sum(1)]                 # ax + by + cz + d = 0
        K = planes[:, :, None] * planes[:, None, :]         # quadrique de chaque face
        self.Q = np.zeros((len(self.V), 4, 4))
        for k in range(3):
            np.add.at(self.Q, self.F[:, k], K)
        for i, j in self.border:     # plan perpendiculaire à la face, contenant l'arête
            f, = self.vf[i] & self.vf[j]
            m = np.cross(self.V[j] - self.V[i], n[f])
            p = np.append(m, -m @ self.V[i]) / np.linalg.norm(m)
            self.Q[[i, j]] += BORDER_WEIGHT * np.outer(p, p)

    def edge_cost(self, a, b):
        """Erreur vT(Qa+Qb)v minimale sur les positions candidates de v.
        Renvoie (coût, s gardé, t supprimé, nouvelle position p de s)."""
        Q = self.Q[a] + self.Q[b]
        cands = [self.V[a], self.V[b]]
        if not HALF_EDGE:    # minimum de vT Q v : A v = -c (pinv : stable si zone plane)
            A, c, m = Q[:3, :3], Q[:3, 3], (self.V[a] + self.V[b]) / 2
            cands.append(m - np.linalg.pinv(A, rcond=1e-3) @ (A @ m + c))
        err = [np.append(p, 1) @ Q @ np.append(p, 1) for p in cands]
        i = int(np.argmin(err))
        return (err[i], b, a, cands[i]) if i == 1 else (err[i], a, b, cands[i])

    # 4. Choix de l'arête ---------------------------------------------------
    def push(self, a, b):
        e = (min(a, b), max(a, b))
        self.candidates[e] = self.edge_cost(*e)
        heapq.heappush(self.heap, (self.candidates[e][0], e))

    def update(self, v):
        """Après un collapse en v : recalcule les arêtes des faces autour de v."""
        for e in {tuple(sorted(e)) for f in self.vf[v] for e in combinations(self.F[f], 2)}:
            self.push(*e)

    def pop_best_edge(self):
        """Contraction valide la moins chère : (s, t, p, coût), t sera fusionné dans s."""
        while self.heap:
            cost, e = heapq.heappop(self.heap)
            if e in self.candidates and self.candidates[e][0] == cost:  # sinon périmée
                _, s, t, p = self.candidates.pop(e)
                if self.collapse_ok(s, t, p):
                    return s, t, p, cost
        return None

    def collapse_ok(self, s, t, p):
        shared = self.vf[s] & self.vf[t]                 # faces (s,t,vl) et (t,s,vr)
        if not 0 < len(shared) <= 2:
            return False                                 # arête disparue ou non-manifold
        opposite = set(self.F[list(shared)].ravel()) - {s, t}
        if self.ring(s) & self.ring(t) != opposite:
            return False                                 # condition de lien : pincement
        if len(shared) == 2 and self.on_border(s) and self.on_border(t):
            return False                                 # relierait deux bords
        fan = (self.vf[s] | self.vf[t]) - shared         # faces qui restent
        merged = [frozenset(self.F[f]) - {t} | {s} for f in fan]
        if not fan or len(set(merged)) < len(merged):
            return False                                 # composante détruite, tétraèdre
        for f in fan:                                    # s et t vont en p
            old = self.V[self.F[f]]
            new = np.where(np.isin(self.F[f], (s, t))[:, None], p, old)
            if np.cross(*(old[1:] - old[0])) @ np.cross(*(new[1:] - new[0])) <= 0:
                return False                             # face retournée
        return True


if __name__ == '__main__':
    pm = ProgressiveMesh()
    pm.prepare(sys.argv[1] if len(sys.argv) > 1 else 'example/suzanne.obj')
    print(f'{len(pm.V)} sommets, {len(pm.F)} faces, '
          f'{len(pm.candidates)} arêtes dont {len(pm.border)} de bord')
    print('5 meilleures candidates (rien n\'est contracté avant l\'étape 5) :')
    for _ in range(5):
        s, t, p, cost = pm.pop_best_edge()
        print(f'  {t} -> {s}   coût {max(cost, 0):.1e}   longueur {np.linalg.norm(pm.V[s] - pm.V[t]):.4f}')
