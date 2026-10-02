#!/usr/bin/env python3

import obja
import numpy as np


class Collapse:
    """
    Informations nécessaires pour reconstruire.
    """

    def __init__(self, vertex_id, vertex_position, faces):
        self.vertex_id = vertex_id
        self.vertex_position = np.copy(vertex_position)
        self.faces = [f.clone() for f in faces]


class ProgressiveMesh(obja.Model):

    def __init__(self):
        super().__init__()

        self.deleted_faces = set()
        self.history = []

    # -----------------------------
    # Géométrie
    # -----------------------------

    def edge_length(self, a, b):
        return np.linalg.norm(
            self.vertices[a] - self.vertices[b]
        )

    def collect_edges(self):

        edges = set()

        for face in self.faces:

            if not face.visible:
                continue

            edges.add(tuple(sorted((face.a, face.b))))
            edges.add(tuple(sorted((face.b, face.c))))
            edges.add(tuple(sorted((face.c, face.a))))

        return list(edges)

    def shortest_edge(self):

        edges = self.collect_edges()

        if len(edges) == 0:
            return None

        best_edge = None
        best_cost = float("inf")

        for edge in edges:

            a, b = edge

            cost = self.edge_length(a, b)

            if cost < best_cost:
                best_cost = cost
                best_edge = edge

        return best_edge

    # -----------------------------
    # Collapse simple
    # -----------------------------

    def collapse_edge(self, vs, vt):

        removed_faces = []

        for face in self.faces:

            if not face.visible:
                continue

            verts = [face.a, face.b, face.c]

            if vs not in verts:
                continue

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
                removed_faces
            )
        )

    # -----------------------------
    # Décimation
    # -----------------------------

    def simplify(self, target_ratio=0.1):

        initial_faces = len(self.faces)

        target_faces = max(
            1,
            int(initial_faces * target_ratio)
        )

        while self.count_visible_faces() > target_faces:

            edge = self.shortest_edge()

            if edge is None:
                break

            vs, vt = edge

            self.collapse_edge(vs, vt)

    def count_visible_faces(self):

        count = 0

        for face in self.faces:
            if face.visible:
                count += 1

        return count

    # -----------------------------
    # Export OBJA
    # -----------------------------

    def write_obja(self, output):

        output_model = obja.Output(
            output,
            random_color=True
        )

        used_vertices = set()

        for face in self.faces:

            if not face.visible:
                continue

            used_vertices.add(face.a)
            used_vertices.add(face.b)
            used_vertices.add(face.c)

        for v_id in sorted(used_vertices):

            output_model.add_vertex(
                v_id,
                self.vertices[v_id]
            )

        face_counter = 0

        for face in self.faces:

            if not face.visible:
                continue

            output_model.add_face(
                face_counter,
                face
            )

            face_counter += 1

        self.history.reverse()

        next_face_index = face_counter

        for collapse in self.history:

            output_model.add_vertex(
                collapse.vertex_id,
                collapse.vertex_position
            )

            for face in collapse.faces:

                output_model.add_face(
                    next_face_index,
                    face
                )

                next_face_index += 1


def main():

    model = ProgressiveMesh()

    model.parse_file("example/suzanne.obj")

    model.simplify(
        target_ratio=0.1
    )

    with open(
        "example/suzanne_pm.obja",
        "w"
    ) as f:
        model.write_obja(f)