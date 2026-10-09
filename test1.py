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
    def collect_edge_neighborhood(self, a, b):
        """
        Collecte le voisinage local de l'edge collapse (faces autour de a et b).
        """
        voisinage = []
        for face in self.faces:
            if not face.visible:
                continue
            verts = [face.a, face.b, face.c]
            if a in verts or b in verts:
                voisinage.append(face.clone())
        return voisinage
    
    def edge_length(self, a, b):
        """
        Calcule le cout estime du collapse de l'arete (a, b).

        D'apres Hoppe et al., Section 4.3:
            E = E_dist + E_spring

        E_dist mesure la deformation geometrique locale.
        E_spring penalise les aretes trop longues.
        """
        # 1. Construire le voisinage local de l'edge collapse (faces autour de a et b).
        voisinage = self.collect_edge_neighborhood(a, b)
        # 2. Choisir la position candidate du sommet conserve apres le collapse.
        if len(voisinage) == 0:
            return 0.0
        sommet_a = self.vertices[a]
        sommet_b = self.vertices[b]
        position_candidate = (sommet_a + sommet_b) / 2.0

        # 3. Evaluer l'erreur geometrique E_dist entre le maillage actuel
        #    et le maillage obtenu apres la contraction.
        e_dist = 0.0
        for face in voisinage:
            v1 = self.vertices[face.a]
            v2 = self.vertices[face.b]
            v3 = self.vertices[face.c]
            normale = np.cross(v2 - v1, v3 - v1)
            norme =  np.linalg.norm(normale)
            if norme == 0:
                continue
            normale /= norme
            distance = np.dot(position_candidate - v1, normale)
            e_dist += distance**2
        # 4. Ajouter, si elle est utilisee, l'energie de regularisation E_spring.
        e_spring = np.linalg.norm(sommet_a - sommet_b)
        # 5. Retourner E = E_dist + E_spring comme priorite du collapse.
        cout_total = e_dist + e_spring
        # La longueur seule ne doit donc plus etre le cout final.
        return cout_total


def collect_edges(self):
    """
    Construit les candidats d'edge collapse et les structures de voisinage.
    Hoppe et al. [9], Sections 4.1 et 4.3:
    les collapses candidats doivent etre analyses localement,
    puis evalues selon leur cout et leur legalite topologique.
    """
    # Ensemble des aretes uniques du maillage.
    edges = set()
    # Faces adjacentes a chaque arete.
    self.edge_faces = {}
    # Sommets voisins de chaque sommet.
    self.vertex_neighbors = {
        vertex_id: set()
        for vertex_id in range(len(self.vertices))
    }
    for face in self.faces:
        # Une face invisible ne participe plus au maillage courant.
        if not face.visible:
            continue
        # Les trois aretes de la face.
        face_edges = [
            (face.a, face.b),
            (face.b, face.c),
            (face.c, face.a)
        ]
        for first_vertex, second_vertex in face_edges:
            # Une arete est representee dans un ordre unique.
            edge = tuple(sorted((first_vertex, second_vertex)))
            # Ignorer une arete degener ee.
            if edge[0] == edge[1]:
                continue
            # Ajouter l'arete aux candidats.
            edges.add(edge)
            # Enregistrer la face adjacente a cette arete.
            self.edge_faces.setdefault(edge, set()).add(face)
            # Construire le voisinage des sommets.
            self.vertex_neighbors[first_vertex].add(second_vertex)
            self.vertex_neighbors[second_vertex].add(first_vertex)
    # Retourner toutes les aretes candidates uniques.
    return list(edges)

def shortest_edge(self):
    """
    Selectionne le collapse ayant le plus petit cout estime.
    Hoppe et al. [9], Section 4.3:
    les transformations candidates sont classees dans une priority queue
    selon leur cout E.
    """
    import heapq
    edges = self.collect_edges()
    if not edges:
        return None
    priority_queue = []
    for edge in edges:
        first_vertex, second_vertex = edge
        cost = self.edge_length(first_vertex, second_vertex)
        heapq.heappush(
            priority_queue,
            (cost, edge)
        )
    while priority_queue:
        cost, edge = heapq.heappop(priority_queue)
        first_vertex, second_vertex = edge
        # Le test de legalite complet sera ajoute ici.
        return edge
    return None

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