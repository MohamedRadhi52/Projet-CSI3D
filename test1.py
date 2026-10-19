#!/usr/bin/env python3
"""Transformation OBJ -> OBJA par QEM, dans le programme initial du projet.

Utilisation : python test1.py suzanne --ratio 0.1 --samples 2000

QEM de Garland-Heckbert, variante de demi-contractions : on évalue les deux
extrémités de chaque arête et on conserve celle de plus faible coût admissible.
Pas d'optimisation continue de position, ni d'agrégation de sommets non reliés.
Les sommets ne bougent pas : le parseur et le visualiseur OBJA restent inchangés.
La sortie contient la base et les inverses des contractions. Le format texte
progressif peut être plus lourd que l'OBJ ; aucune compression binaire ajoutée.
"""

import argparse
import csv
import heapq
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import obja

class Collapse:
    """
    Informations nécessaires pour reconstruire.
    """

    def __init__(self, vertex_id, vertex_position, faces, target_id,
                 target_position, faces_before, quadrics_before):
        self.vertex_id = vertex_id
        self.vertex_position = np.copy(vertex_position)
        self.faces = [f.clone() for f in faces]
        self.target_id = target_id
        self.target_position = np.copy(target_position)
        # Identifiants stables et copies AVANT modification, même pour les
        # faces qui restent visibles après la contraction.
        self.faces_before = {i: f.clone() for i, f in faces_before.items()}
        self.quadrics_before = quadrics_before.copy()



class ProgressiveMesh(obja.Model):

    def __init__(self):
        super().__init__()

        self.deleted_faces = set()
        self.history = []
        self.quadrics = None


    def collect_edges(self):
        """Retourne les arêtes uniques des faces visibles, dans un ordre stable."""

        # L'ensemble élimine les arêtes communes à plusieurs triangles.
        edges = set()

        for face in self.faces:

            if not face.visible:
                continue

            # Trier les extrémités identifie (a, b) et (b, a) comme la même arête.
            edges.add(tuple(sorted((face.a, face.b))))
            edges.add(tuple(sorted((face.b, face.c))))
            edges.add(tuple(sorted((face.c, face.a))))

        return sorted(edges)


    def face_plane(self, face):
        """Retourne [a, b, c, d] avec une normale unitaire, ou None si dégénérée."""
        p0, p1, p2 = (self.vertices[i] for i in (face.a, face.b, face.c))
        edge1 = p1 - p0
        edge2 = p2 - p0
        normal = np.cross(edge1, edge2)
        length = np.linalg.norm(normal)
        # Seuil relatif : indépendant de l'unité utilisée pour les coordonnées.
        scale = np.linalg.norm(edge1) * np.linalg.norm(edge2)
        if length <= 1e-12 * scale:
            return None
        normal = normal / length
        d = -np.dot(normal, p0)
        return np.array([normal[0], normal[1], normal[2], d])


    def compute_initial_quadrics(self):
        """Construit Q_v = somme des p p^T des faces visibles incidentes.

        À calculer avant la simplification. Pour QEM, les contractions devront
        ensuite additionner ces quadriques historiques, pas les recalculer
        à partir des seules faces restantes.
        """
        quadrics = np.zeros((len(self.vertices), 4, 4))
        skipped = 0
        for face in self.faces:
            if not face.visible:
                continue
            plane = self.face_plane(face)
            if plane is None:
                skipped += 1
                continue
            # Produit extérieur : matrice 4 x 4, et non produit scalaire.
            face_quadric = np.outer(plane, plane)
            for vertex_id in (face.a, face.b, face.c):
                quadrics[vertex_id] += face_quadric
        return quadrics, skipped


    def qem_cost(self, vs, vt):
        """Erreur de vs -> vt, évaluée à la position inchangée de vt."""
        if self.quadrics is None:
            self.quadrics, _ = self.compute_initial_quadrics()
        quadric = self.quadrics[vs] + self.quadrics[vt]
        position = np.append(self.vertices[vt], 1.0)
        # Une erreur quadratique est positive ; arrondis flottants près de zéro.
        return max(0.0, float(position @ quadric @ position))


    def collapse_edge(self, vs, vt):

        if vs == vt or tuple(sorted((vs, vt))) not in self.collect_edges():
            raise ValueError("La contraction doit porter sur une arête visible.")

        if self.quadrics is None:
            self.quadrics, _ = self.compute_initial_quadrics()
        quadrics_before = self.quadrics[[vs, vt]].copy()

        removed_faces = []
        faces_before = {}

        for face_id, face in enumerate(self.faces):

            if not face.visible:
                continue

            verts = [face.a, face.b, face.c]

            if vs not in verts:
                continue

            faces_before[face_id] = face.clone()

            new_verts = [
                vt if v == vs else v
                for v in verts
            ]

            if len(set(new_verts)) < 3:

                removed_faces.append(face.clone())
                face.visible = False

            else:

                face.a = new_verts[0]
                face.b = new_verts[1]
                face.c = new_verts[2]

        self.history.append(
            Collapse(
                vs,
                self.vertices[vs],
                removed_faces,
                vt,
                self.vertices[vt],
                faces_before,
                quadrics_before
            )
        )
        # Conserver l'erreur historique, même lorsque des faces disparaissent.
        self.quadrics[vt] += self.quadrics[vs]
        self.quadrics[vs] = 0.0


    def mesh_context(self):
        """Voisinages actuels pour les coûts locaux et les contrôles communs."""
        incident = [set() for _ in self.vertices]
        adjacent = [set() for _ in self.vertices]
        edge_faces = {}
        visible = [i for i, f in enumerate(self.faces) if f.visible]
        ids_array = np.array([(self.faces[i].a, self.faces[i].b, self.faces[i].c)
                              for i in visible], dtype=int).reshape(-1, 3)
        triangles = np.array(self.vertices)[ids_array]
        cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        scale = np.linalg.norm(triangles[:, 1]-triangles[:, 0], axis=1) * np.linalg.norm(triangles[:, 2]-triangles[:, 0], axis=1)
        unit = np.divide(cross, lengths[:, None], out=np.zeros_like(cross),
                         where=(lengths > 1e-12*scale)[:, None])
        normals = dict(zip(visible, unit))
        for i in visible:
            f = self.faces[i]
            ids = (f.a, f.b, f.c)
            for v in ids:
                incident[v].add(i)
            for a, b in ((f.a, f.b), (f.b, f.c), (f.c, f.a)):
                adjacent[a].add(b)
                adjacent[b].add(a)
                edge_faces.setdefault(tuple(sorted((a, b))), []).append(i)
        boundary = {v for edge, faces in edge_faces.items() if len(faces) == 1 for v in edge}
        return incident, adjacent, edge_faces, normals, boundary


    def legal_collapse(self, vs, vt, context):
        """Lien local, bords, triangles aplatis/inversés et faces dupliquées.

        Ces tests communs ne détectent pas toutes les auto-intersections globales.
        """
        incident, adjacent, edge_faces, normals, boundary = context
        shared = edge_faces.get(tuple(sorted((vs, vt))), [])
        if len(shared) not in (1, 2):
            return False
        opposite = {v for i in shared for v in (self.faces[i].a, self.faces[i].b, self.faces[i].c)
                    if v not in (vs, vt)}
        if adjacent[vs] & adjacent[vt] != opposite:
            return False
        if vs in boundary and (vt not in boundary or len(shared) != 1):
            return False
        keys = set()
        surviving = 0
        for i in incident[vs] | incident[vt]:
            f = self.faces[i]
            ids = tuple(vt if v == vs else v for v in (f.a, f.b, f.c))
            if len(set(ids)) < 3:
                continue
            surviving += 1
            key = tuple(sorted(ids))
            if key in keys:
                return False
            keys.add(key)
            if vs in (f.a, f.b, f.c):
                x, y, z = (self.vertices[v] for v in ids)
                normal = np.cross(y - x, z - x)
                scale = np.linalg.norm(y - x) * np.linalg.norm(z - x)
                if np.linalg.norm(normal) <= 1e-12 * scale or np.dot(normal, normals[i]) <= 0:
                    return False
        return surviving > 0


    def count_visible_faces(self):

        count = 0

        for face in self.faces:
            if face.visible:
                count += 1

        return count


    def write_obja(self, output):

        """Écrit la base simplifiée puis les opérations inverses des contractions.

        Version pédagogique : toutes les positions sont écrites au départ.
        Valable pour nos demi-contractions, qui ne déplacent aucun sommet.
        """
        output_model = obja.Output(output, random_color=False)
        output.write("# Sommets : positions conservees pendant les contractions\n")
        for vertex_id, position in enumerate(self.vertices):
            output_model.add_vertex(vertex_id, position)

        output.write("# Base simplifiee\n")
        for face_id, face in enumerate(self.faces):
            if face.visible:
                output_model.add_face(face_id, face)

        # reversed parcourt sans modifier ni consommer l'historique.
        for collapse in reversed(self.history):
            output.write(f"# Annulation de {collapse.vertex_id} -> {collapse.target_id}\n")
            for face_id, old_face in collapse.faces_before.items():
                if face_id in output_model.face_mapping:
                    # Face survivante : restaurer ses anciens indices.
                    output_model.edit_face(face_id, old_face)
                else:
                    # Face disparue : la réintroduire, avec son identité d'origine.
                    output_model.add_face(face_id, old_face)


    def simplify(self, target_ratio=0.1):
        """QEM uniquement, avec quadriques historiques et tas de coûts locaux.

        Contrôles de lien, de bord, de doublons et d'orientation. Ils ne
        constituent pas une détection globale de toutes les auto-intersections.
        La cible n'est pas franchie ; l'arrêt peut rester au-dessus de celle-ci.
        """
        if not math.isfinite(target_ratio) or not 0 < target_ratio <= 1:
            raise ValueError("Le ratio doit appartenir à ]0,1].")
        if self.quadrics is None:
            self.quadrics, _ = self.compute_initial_quadrics()
        self.target_faces = max(1, math.ceil(self.count_visible_faces()*target_ratio))
        context = self.mesh_context()
        queue, tokens, serial = [], {}, 0

        def enqueue(a, b):
            nonlocal serial
            for vs, vt in ((a, b), (b, a)):
                serial += 1
                tokens[vs, vt] = serial
                if self.legal_collapse(vs, vt, context):
                    cost = self.qem_cost(vs, vt)
                    if np.isfinite(cost):
                        heapq.heappush(queue, (cost, serial, vs, vt))

        def rebuild():
            queue.clear()
            tokens.clear()
            for a, b in sorted(context[2]):
                enqueue(a, b)

        rebuild()
        self.stop_reason = "target_reached"
        while self.count_visible_faces() > self.target_faces:
            selected = None
            while queue:
                _, token, vs, vt = heapq.heappop(queue)
                if tokens.get((vs, vt)) != token or not self.legal_collapse(vs, vt, context):
                    continue
                removed = len(context[2][tuple(sorted((vs, vt)))])
                if self.count_visible_faces()-removed < self.target_faces:
                    continue
                selected = vs, vt
                break
            if selected is None:
                self.stop_reason = "no_legal_candidate_at_target_budget"
                break
            vs, vt = selected
            touched = context[1][vs] | context[1][vt] | {vs, vt}
            self.collapse_edge(vs, vt)
            context = self.mesh_context()
            touched |= context[1][vt]
            edges = {tuple(sorted((v, w))) for v in touched for w in context[1][v]}
            for a, b in sorted(edges):
                enqueue(a, b)
            if len(queue) > 30*max(1, len(context[2])):
                rebuild()



def sample_surface(triangles, count, seed):
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                    triangles[:, 2]-triangles[:, 0]), axis=1)
    if not len(areas) or areas.sum() <= 0:
        raise ValueError("Surface sans aire positive.")
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles), count, p=areas/areas.sum())]
    r, s = np.sqrt(rng.random(count))[:, None], rng.random(count)[:, None]
    return (1-r)*selected[:, 0]+r*(1-s)*selected[:, 1]+r*s*selected[:, 2]


def surface_distances(points, triangles):
    """Minimum exact sur les triangles cibles, pour chaque point échantillonné.

    Lots sur les points ET les triangles pour borner la mémoire sur Bunny.
    """
    result = np.empty(len(points))
    for start in range(0, len(points), 32):
        p = points[start:start+32, None, :]
        closest = np.full(len(p), np.inf)
        for offset in range(0, len(triangles), 4096):
            tri = triangles[offset:offset+4096]
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            normal = np.cross(b-a, c-a)
            normal2 = np.sum(normal*normal, axis=1)
            best2 = np.full((len(p), len(tri)), np.inf)
            inside = np.ones(best2.shape, dtype=bool)
            for u, v in ((a, b), (b, c), (c, a)):
                edge, delta = v-u, p-u
                edge2 = np.sum(edge*edge, axis=1)
                t = np.divide(np.sum(delta*edge, axis=2), edge2,
                              out=np.zeros(best2.shape), where=edge2 > 0)
                residual = delta-np.clip(t, 0, 1)[:, :, None]*edge
                best2 = np.minimum(best2, np.sum(residual*residual, axis=2))
                inside &= np.sum(np.cross(edge, delta)*normal, axis=2) >= 0
            height = np.sum((p-a)*normal, axis=2)
            plane2 = np.divide(height*height, normal2,
                               out=np.full(best2.shape, np.inf), where=normal2 > 0)
            best2 = np.minimum(best2, np.where(inside, plane2, np.inf))
            closest = np.minimum(closest, np.min(best2, axis=1))
        result[start:start+len(p)] = np.sqrt(closest)
    return result


def evaluate_surfaces(original, base, samples=2000, seed=42):
    if samples < 1 or seed < 0:
        raise ValueError("Échantillonnage invalide.")
    diagonal = float(np.linalg.norm(np.ptp(original.reshape(-1, 3), axis=0)))
    if diagonal <= 0:
        raise ValueError("Diagonale nulle.")
    origin = original.reshape(-1, 3).min(axis=0)
    original, base = (original-origin)/diagonal, (base-origin)/diagonal
    forward = surface_distances(sample_surface(original, samples, seed), base)
    backward = surface_distances(sample_surface(base, samples, seed+1), original)
    both = np.concatenate((forward, backward))
    rmse = float(np.sqrt(np.mean(both**2)))
    return {"normalization_diagonal": diagonal,
            "mean_original_to_base_normalized": float(forward.mean()),
            "mean_base_to_original_normalized": float(backward.mean()),
            "mean_symmetric_normalized": float(both.mean()),
            "rmse_symmetric_normalized": rmse,
            "rmse_symmetric": rmse*diagonal,
            "p95_symmetric_normalized": float(np.percentile(both, 95)),
            "sampled_max_symmetric_normalized": float(both.max())}


def main(argv=None):
    # Étape 1 : lire les paramètres de la commande.
    parser = argparse.ArgumentParser(description="QEM et reconstruction progressive OBJA.")
    parser.add_argument("example", nargs="?", default="suzanne")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent/"example")
    args = parser.parse_args(argv)

    # Étape 2 : charger l'OBJ et garder l'original pour les comparaisons.
    source = Path(args.example)
    if not source.is_file():
        name = source if source.suffix else source.with_suffix(".obj")
        source = Path(__file__).resolve().parent/"example"/name
    model = ProgressiveMesh()
    model.parse_file(source)
    original_vertices = np.array(model.vertices, dtype=float)
    original_faces = [(f.a, f.b, f.c) for f in model.faces if f.visible]
    original_triangles = original_vertices[np.array(original_faces)]

    # Étape 3 : calculer les quadriques, simplifier et mesurer le temps.
    preparation_start = perf_counter()
    model.quadrics, skipped = model.compute_initial_quadrics()
    model.simplify(args.ratio)
    simplification_seconds = perf_counter()-preparation_start
    base_faces = [(f.a, f.b, f.c) for f in model.faces if f.visible]
    base_triangles = np.array(model.vertices)[np.array(base_faces)]

    # Étape 4 : écrire la base simplifiée et les raffinements dans l'OBJA.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{source.stem}_qem_r{args.ratio:g}"
    obja_path = args.output_dir/f"{stem}.obja"
    with obja_path.open("w", encoding="utf-8") as output:
        model.write_obja(output)

    # Étape 5 : relire l'OBJA et vérifier la reconstruction exacte de l'original.
    decoded = obja.Model()
    decoded.parse_file(obja_path)
    reconstructed = sorted((f.a, f.b, f.c) for f in decoded.faces if f.visible)
    exact = np.array_equal(np.array(decoded.vertices), original_vertices) and reconstructed == sorted(original_faces)
    if not exact:
        raise RuntimeError("La reconstruction de l'OBJA diffère du modèle initial.")

    # Étape 6 : mesurer l'erreur entre la surface originale et la base simplifiée.
    evaluation_start = perf_counter()
    geometry = evaluate_surfaces(original_triangles, base_triangles, args.samples, args.seed)
    evaluation_seconds = perf_counter()-evaluation_start

    # Étape 7 : enregistrer les métriques JSON/CSV et afficher le résumé.
    row = {"method": "qem", "placement": "endpoints", "model": source.name,
           "ratio_requested": args.ratio, "ratio_achieved": len(base_faces)/len(original_faces),
           "initial_vertices": len(original_vertices), "initial_faces": len(original_faces),
           "target_faces": model.target_faces, "base_faces": len(base_faces),
           "collapses": len(model.history), "degenerate_faces_in_quadrics": skipped,
           "stop_reason": model.stop_reason, "samples_per_direction": args.samples, "seed": args.seed,
           "simplification_seconds": simplification_seconds, "evaluation_seconds": evaluation_seconds,
           "input_bytes": source.stat().st_size, "obja_bytes": obja_path.stat().st_size,
           "reconstruction_exact": bool(exact), **geometry}
    json_path = args.output_dir/f"{stem}_metrics.json"
    csv_path = args.output_dir/f"{stem}_metrics.csv"
    json_path.write_text(json.dumps(row, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"QEM : {len(original_faces)} -> {len(base_faces)} faces, {len(model.history)} contractions")
    print(f"Arrêt : {model.stop_reason} ; reconstruction OBJA exacte : oui")
    print(f"RMSE : {geometry['rmse_symmetric_normalized']:.6%} de la diagonale")
    print(f"P95 : {geometry['p95_symmetric_normalized']:.6%} de la diagonale")
    for output in (obja_path, json_path, csv_path):
        print(output.resolve())
    return row


if __name__ == "__main__":
    main()
