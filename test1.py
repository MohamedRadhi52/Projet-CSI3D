#!/usr/bin/env python3

import heapq
import sys
import numpy as np
import obja


class Collapse:
    def __init__(self, vs, vs_pos, vt, vt_pos, cost, removed_faces):
        self.vertex_id = vs
        self.vertex_position = np.copy(vs_pos)
        self.target_vertex = vt
        self.target_position = np.copy(vt_pos)
        self.collapse_cost = cost
        # Sauvegarde une copie exacte des faces supprimées AVANT modification
        self.faces = [f.clone() for f in removed_faces]


class ProgressiveMesh(obja.Model):

    def __init__(self):
        super().__init__()
        self.history = []

    def collect_edges(self):
        edges = set()
        for face in self.faces:
            if not face.visible:
                continue
            verts = [face.a, face.b, face.c]
            for i in range(3):
                u, v = verts[i], verts[(i + 1) % 3]
                edges.add((min(u, v), max(u, v)))
        return edges

    def edge_length_qem(self, a, b):
        length = np.linalg.norm(self.vertices[a] - self.vertices[b])
        degree_a = sum(1 for f in self.faces if f.visible and a in (f.a, f.b, f.c))
        degree_b = sum(1 for f in self.faces if f.visible and b in (f.a, f.b, f.c))
        return length * (1.0 + 0.1 * (degree_a + degree_b))

    def build_heap(self):
        heap = []
        for a, b in self.collect_edges():
            cost = self.edge_length_qem(a, b)
            heapq.heappush(heap, (cost, (a, b)))
        return heap

    def compute_barycenter(self, vs, vt):
        neighbors = set()
        for face in self.faces:
            if not face.visible:
                continue
            verts = [face.a, face.b, face.c]
            if vs in verts or vt in verts:
                for v in verts:
                    if v != vs and v != vt:
                        neighbors.add(v)

        points = [self.vertices[vs], self.vertices[vt]]
        for n in neighbors:
            points.append(self.vertices[n])

        return np.mean(points, axis=0)

    def is_valid_edge(self, u, v):
        for face in self.faces:
            if not face.visible:
                continue
            verts = [face.a, face.b, face.c]
            if u in verts and v in verts:
                return True
        return False

    def collapse_edge_improved(self, vs, vt, cost):
        removed_faces = []

        old_vs_pos = np.copy(self.vertices[vs])
        old_vt_pos = np.copy(self.vertices[vt])

        # 1. Identifier et cloner les faces qui vont être supprimées (dégénérées)
        for face in self.faces:
            if not face.visible:
                continue

            verts = [face.a, face.b, face.c]
            if vs in verts:
                new_verts = [vt if v == vs else v for v in verts]
                if len(set(new_verts)) < 3:
                    # Copie de l'état ORIGINAL de la face avant suppression
                    removed_faces.append(face.clone())
                    face.visible = False

        # 2. Appliquer le déplacement géométrique
        self.vertices[vt] = self.compute_barycenter(vs, vt)

        # 3. Mettre à jour les faces restantes
        for face in self.faces:
            if not face.visible:
                continue
            if face.a == vs: face.a = vt
            if face.b == vs: face.b = vt
            if face.c == vs: face.c = vt

        # 4. Enregistrer dans l'historique
        self.history.append(
            Collapse(vs, old_vs_pos, vt, old_vt_pos, cost, removed_faces)
        )

    def simplify2(self, target_ratio=0.1, max_cost=5.0):
        initial_faces = self.count_visible_faces()
        target_faces = max(1, int(initial_faces * target_ratio))

        heap = self.build_heap()

        while self.count_visible_faces() > target_faces:
            if not heap:
                break

            cost, (vs, vt) = heapq.heappop(heap)

            if cost > max_cost:
                break

            if not self.is_valid_edge(vs, vt):
                continue

            self.collapse_edge_improved(vs, vt, cost)
            heap = self.build_heap()

        print(f"Faces restantes : {self.count_visible_faces()} / {initial_faces}")

    def count_visible_faces(self):
        return sum(1 for face in self.faces if face.visible)

    def write_obja_v2(self, output):
        output_model = obja.Output(output, random_color=True)

        # 1. Écriture du maillage de base (simplifié au maximum)
        visible_faces = [face for face in self.faces if face.visible]
        used_vertices = set()

        for face in visible_faces:
            used_vertices.add(face.a)
            used_vertices.add(face.b)
            used_vertices.add(face.c)

        # Sommets de la base
        for v_id in sorted(used_vertices):
            output_model.add_vertex(v_id, self.vertices[v_id])

        # Faces de la base
        face_counter = 0
        for face in visible_faces:
            output_model.add_face(face_counter, face)
            face_counter += 1

        # 2. Section maillage progressif (Inversion de l'historique)
        output.write(f"# progressive_mesh {len(self.history)} collapses\n")

        for collapse in reversed(self.history):
            vt = collapse.target_vertex
            vs = collapse.vertex_id

            # S'assurer que le sommet cible (vt) existe dans le mapping avant de le modifier
            if vt not in output_model.vertex_mapping:
                output_model.add_vertex(vt, collapse.target_position)
            else:
                output_model.edit_vertex(vt, collapse.target_position)

            # Restauration du sommet supprime (vs)
            if vs not in output_model.vertex_mapping:
                output_model.add_vertex(vs, collapse.vertex_position)

            # Restauration des faces
            for face in collapse.faces:
                # Vérification de sécurité pour les sommets de la face
                for v_id in (face.a, face.b, face.c):
                    if v_id not in output_model.vertex_mapping:
                        output_model.add_vertex(v_id, self.vertices[v_id])

                output_model.add_face(face_counter, face)
                face_counter += 1


def main():
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = "example/suzanne.obj"
        output_file = "example/suzanne_pm.obja"

    model = ProgressiveMesh()
    model.parse_file(input_file)
    model.simplify2(target_ratio=0.1, max_cost=5.0)

    with open(output_file, "w") as f:
        model.write_obja_v2(f)


if __name__ == "__main__":
    main()