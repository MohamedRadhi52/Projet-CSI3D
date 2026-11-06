#!/usr/bin/env python3
"""
Progressive Mesh : compression progressive d'un maillage au format OBJA.

  1. load()             chargement + voisinage (sommet -> faces)
  2. collect_edges()    arêtes uniques + arêtes de bord
  3. edge_cost()        coût QEM (Garland & Heckbert 97), pondéré par la saillance
  4. pop_best_edge()    tas de priorité + test de validité
  5-7. collapse_edge()  contraction, suppression des faces plates, vsplit mémorisé
  8-9. simplify()       boucle jusqu'au ratio voulu ou plus de contraction valide
  10-11. write_obja()   maillage de base M0 puis les vsplit dans l'ordre inverse
  decode()              décompression : rejoue le .obja

Critère perceptif (Limper et al. 2016) : compute_saliency() donne S entre 0 et 1.
  LAMBDA > 0  : coût = QEM x (1 + LAMBDA x S)   (pondération)
  PROTECT > 0 : la fraction PROTECT des sommets les plus saillants est protégée,
                seuil relâché quand plus rien n'est contractable (méthode de l'article)

Usage : python3 progressive_mesh.py example/bunny.obj [sortie.obja] [ratio] [lambda]
"""
import heapq
import sys
from itertools import combinations

import numpy as np

import obja

HALF_EDGE = True     # True : s garde sa position (vsplit sans 'ev'), False : position optimale
BORDER_WEIGHT = 1.0  # plans de bord (Garland & Heckbert §6) ; 0 = les trous s'agrandissent
LAMBDA = 0.0         # poids de la saillance dans le coût (0 = QEM seul)
PROTECT = 0.0        # fraction de sommets protégés par un seuil de saillance (0 = aucun)


class ProgressiveMesh(obja.Model):

    def prepare(self, path):
        self.load(path)                       # 1
        edges = self.collect_edges()          # 2
        self.compute_quadrics()               # 3
        self.S = self.compute_saliency() if LAMBDA or PROTECT else np.zeros(len(self.V))
        self.protect = PROTECT
        self.threshold = np.quantile(self.S, 1 - PROTECT) if PROTECT else np.inf
        self.heap, self.candidates = [], {}   # 4 : arête -> (coût, s, t, p) à jour
        for a, b in edges:
            self.push(a, b)
        self.history = []                     # 7 : un vsplit par contraction

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

    def compute_saliency(self):
        """Saillance (Limper et al. 2016) : entropie des courbures moyennes des voisins,
        sur 5 rayons, moyennée, lissée une fois, ramenée entre 0 et 1."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        H, area = mean_curvature(self.V, self.F)
        radii = 0.08 * np.sqrt(area.sum()) / 2.0 ** np.arange(5)     # r_max, r_max/2 ... /16
        area[[v for e in self.border for v in e]] = 0                 # courbure peu fiable au bord
        lo, hi = np.percentile(H[np.unique(self.F)], [1, 99])
        b = np.clip(((H - lo) / (hi - lo) * 256).astype(int), 0, 255)  # 256 classes
        E = np.array(self.collect_edges())
        w = np.linalg.norm(self.V[E[:, 0]] - self.V[E[:, 1]], axis=1)
        n = len(self.V)
        G = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), (n, n)).tocsr()
        S = np.zeros(n)
        for start in range(0, n, 256):                                # par paquets de sommets
            rows = range(start, min(n, start + 256))
            D = dijkstra(G, indices=rows, limit=radii[0])             # distances sur la surface
            for row, v in enumerate(rows):
                for r in radii:
                    near = D[row] <= r
                    p = np.bincount(b[near], weights=area[near], minlength=256)
                    p = p[p > 0] / p.sum()
                    S[v] -= (p * np.log2(p)).sum()                    # entropie de Shannon
        smooth, count = S.copy(), np.ones(n)                          # lissage : moyenne avec les voisins
        np.add.at(smooth, E[:, 0], S[E[:, 1]])
        np.add.at(smooth, E[:, 1], S[E[:, 0]])
        np.add.at(count, E.ravel(), 1)
        S, used = smooth / count, np.unique(self.F)
        return np.clip((S - S[used].min()) / np.ptp(S[used]), 0, 1)

    def edge_cost(self, a, b):
        """Erreur vT(Qa+Qb)v minimale sur les positions candidates de v, x (1 + LAMBDA x S).
        Renvoie (coût, s gardé, t supprimé, nouvelle position p de s)."""
        Q = self.Q[a] + self.Q[b]
        cands = [self.V[a], self.V[b]]
        if not HALF_EDGE:    # minimum de vT Q v : A v = -c (pinv : stable si zone plane)
            A, c, m = Q[:3, :3], Q[:3, 3], (self.V[a] + self.V[b]) / 2
            cands.append(m - np.linalg.pinv(A, rcond=1e-3) @ (A @ m + c))
        lost = [self.S[b], self.S[a], max(self.S[a], self.S[b])]   # saillance de ce qui disparaît
        err = [(np.append(p, 1) @ Q @ np.append(p, 1)) * (1 + LAMBDA * s) for p, s in zip(cands, lost)]
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
        if self.S[t] > self.threshold:
            return False                                 # sommet saillant protégé (article)
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

    # 5-7. Contraction + historique -----------------------------------------
    def collapse_edge(self, s, t, p):
        """Fusionne t dans s, placé en p, et mémorise le vsplit qui annule l'opération."""
        shared = self.vf[s] & self.vf[t]                 # 6. faces qui deviennent plates
        moved = [(f, int(np.flatnonzero(self.F[f] == t)[0])) for f in self.vf[t] - shared]
        self.history.append((s, t, self.V[s].copy(), self.V[t].copy(), sorted(shared), moved))
        for f in shared:
            for v in self.F[f]:
                self.vf[v].discard(f)
        for f, corner in moved:
            self.F[f, corner] = s
        self.vf[s] |= self.vf[t]
        self.vf[t] = set()
        self.V[s] = p
        self.Q[s] += self.Q[t]                           # s garde la mémoire de l'erreur
        self.S[s] = max(self.S[s], self.S[t])            # ... et de la saillance
        self.update(s)

    # 8-9. Boucle et critère d'arrêt ----------------------------------------
    def simplify(self, ratio=0.0):
        """Garde ratio x (nombre de sommets), ou s'arrête quand plus rien n'est contractable."""
        n = len(np.unique(self.F))
        target = int(ratio * n)
        while n > target:
            best = self.pop_best_edge()
            if best is None:
                if self.threshold == np.inf:
                    break
                self.protect /= 2                        # article : on relâche le seuil
                self.threshold = np.quantile(self.S, 1 - self.protect) if self.protect * n > 1 else np.inf
                faces = list(set().union(*self.vf))
                for e in np.unique(np.sort(self.F[faces][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0):
                    self.push(*e)
                continue
            self.collapse_edge(*best[:3])
            n -= 1

    # 10-11. Écriture OBJA --------------------------------------------------
    def write_obja(self, file):
        """M0 puis les vsplit du dernier collapse au premier ; 's' = octets envoyés."""
        out = obja.Output(ByteCounter(file))
        faces = sorted(set().union(*self.vf))            # faces de M0
        for v in np.unique(self.F[faces]):
            out.add_vertex(v, self.V[v])
        for f in faces:
            out.add_face(f, obja.Face(*self.F[f]))
        G = self.F.copy()                                # faces au fil de la reconstruction
        step = max(1, len(self.history) // 100)
        for i, (s, t, ps, pt, shared, moved) in enumerate(reversed(self.history)):
            if i % step == 0:
                print('s', out.output.n, file=out.output)
            out.add_vertex(t, pt)
            if not HALF_EDGE:
                out.edit_vertex(s, ps)
            for f, corner in moved:
                G[f, corner] = t
                out.edit_face(f, obja.Face(*G[f]))
            for f in shared:
                out.add_face(f, obja.Face(*G[f]))
        print('s', out.output.n, file=out.output)


class ByteCounter:
    """Fichier qui compte les octets écrits, pour l'instruction 's'."""

    def __init__(self, file):
        self.file, self.n = file, 0

    def write(self, text):
        self.n += len(text.encode())
        return self.file.write(text)


def mean_curvature(V, F):
    """Courbure moyenne signée de chaque sommet (formule des cotangentes)
    et aire de chaque sommet (un tiers de ses triangles)."""
    L, area, normal = np.zeros((len(V), 3)), np.zeros(len(V)), np.zeros((len(V), 3))
    for k in range(3):
        i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        u, w = V[j] - V[i], V[l] - V[i]
        c = np.cross(u, w)
        cot = (u * w).sum(1) / np.maximum(np.linalg.norm(c, axis=1), 1e-12)   # angle en i
        np.add.at(L, j, cot[:, None] / 2 * (V[l] - V[j]))                   # arête opposée (j, l)
        np.add.at(L, l, cot[:, None] / 2 * (V[j] - V[l]))
        np.add.at(area, i, np.linalg.norm(c, axis=1) / 6)
        np.add.at(normal, i, c)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    return -(L * normal).sum(1) / (2 * np.maximum(area, 1e-12)), area


def decode(path):
    """Décompression : rejoue le .obja et donne (octets, sommets, faces) à chaque ligne 's'."""
    V, F = [], []
    with open(path) as file:
        for line in file:
            cmd, *x = line.split() or ['#']
            if cmd == 'v':
                V.append([float(c) for c in x])
            elif cmd == 'ev':
                V[int(x[0]) - 1] = [float(c) for c in x[1:]]
            elif cmd == 'f':
                F.append([int(c) - 1 for c in x])
            elif cmd == 'ef':
                F[int(x[0]) - 1] = [int(c) - 1 for c in x[1:]]
            elif cmd == 's':
                yield int(x[0]), np.array(V), np.array(F)


def check(obj_path, obja_path):
    """Vérifie que la décompression redonne exactement les faces du .obj."""
    m = obja.parse_file(obj_path)
    *_, (_, V, F) = decode(obja_path)
    def key(V, F):
        return sorted(tuple(map(tuple, V[f])) for f in F)
    return key(np.array(m.vertices), np.array([[f.a, f.b, f.c] for f in m.faces])) == key(V, F)


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'example/suzanne.obj'
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace('.obj', '_pm.obja')
    ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    LAMBDA = float(sys.argv[4]) if len(sys.argv) > 4 else LAMBDA
    pm = ProgressiveMesh()
    pm.prepare(src)
    pm.simplify(ratio)
    with open(dst, 'w') as f:
        pm.write_obja(f)
    base = len(set().union(*pm.vf))
    print(f'{dst} : maillage de base de {base} faces + {len(pm.history)} vsplit, '
          f'décompression {"exacte" if check(src, dst) else "FAUSSE"}')
