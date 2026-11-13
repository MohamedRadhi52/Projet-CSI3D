#!/usr/bin/env python3

import argparse
import heapq
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

import obja
import comparaison
import saillance
import qem
import chen


# ======================================================================
# [STRUCTURE PROGRESSIVE MESH] Enregistrement d'un Edge Collapse inversible
# ======================================================================
class Collapse:
    """Un enregistrement de contraction, inversable par vertex split."""

    def __init__(
        self,
        vertex_id,
        vertex_position,
        target_vertex_id,
        target_vertex_position,
        removed_faces,
        modified_faces,
        cost
    ):
        self.vertex_id = vertex_id
        self.vertex_position = np.copy(vertex_position)
        self.target_vertex_id = target_vertex_id
        self.target_vertex_position = np.copy(target_vertex_position)
        self.removed_faces = [
            (face_index, face.clone())
            for face_index, face in removed_faces
        ]
        self.modified_faces = [
            (face_index, old_face.clone(), new_face.clone())
            for face_index, old_face, new_face in modified_faces
        ]
        self.cost = float(cost)

    @property
    def faces(self):
        """Compatibilité avec l'ancien nom utilisé pour les faces supprimées."""
        return [face for _, face in self.removed_faces]


class ProgressiveMesh(obja.Model):
    """
    Progressive mesh de Hoppe basé sur les edge collapses.

    Le maillage courant est le niveau de détail le plus grossier. Chaque
    Collapse peut être annulé par vertex_split(), dans l'ordre inverse.
    """

    def __init__(
        self,
        use_qem=True,
        use_chen=False,
        boundary_weight=10.0,
        saliency_weight=0.0,
        lambda_spring=0.1,
        lambda_rep=0.01,
        lambda_disc=0.1,
    ):
        super().__init__()
        self.history = []
        self.deleted_faces = set()
        self.edge_faces = {}
        self.vertex_neighbors = {}
        self.vertex_faces = {}
        self.reference_vertices = None
        self.lambda_spring = float(lambda_spring)
        self.lambda_rep = float(lambda_rep)
        self.lambda_disc = float(lambda_disc)
        self.use_chen = bool(use_chen)
        self.use_qem = False if self.use_chen else bool(use_qem)
        self.boundary_weight = float(boundary_weight)
        self.saliency_weight = float(saliency_weight)
        self.vertex_saliency = None  # Point d'ancrage pour le collègue (Saillance LCE)
        self.quadrics10 = []
        self._chen_face_normals = {}
        self._chen_unremovable = set()
        self._heap = []
        self._edge_versions = {}
        self._distance_context = {}
        self._pending_candidate = None
        self._pending_edge = None
        self._visible_face_count = None

    # ==================================================================
    # [QEM] Quadriques d'Erreur (Garland & Heckbert 1997 via qem.py)
    # ==================================================================
    def compute_initial_quadrics(self):
        """[QEM] Initialise les quadriques 10-float via le module qem."""
        self.quadrics10 = qem.compute_initial_quadrics(
            self.vertices, self.faces, self.edge_faces, self.boundary_weight
        )

    def qem_cost(self, a, b, position):
        """[QEM] Évalue l'erreur quadrique QEM + saillance facultative via qem.py."""
        if not self.quadrics10 or len(self.quadrics10) != len(self.vertices):
            self.compute_initial_quadrics()
        return qem.eval_edge_cost(
            self.quadrics10, a, b, position, self.saliency_weight, self.vertex_saliency
        )

    # ==================================================================
    # [STRUCTURE COMMUNE & TOPOLOGIE] Incidences arêtes - faces - sommets
    # ==================================================================
    def _visible_face_vertices(self, face):
        return (face.a, face.b, face.c)

    def collect_edges(self):
        """[STRUCTURE] Construit les incidences sommet-arête-face du maillage courant."""
        if self.reference_vertices is None:
            self.reference_vertices = np.asarray(self.vertices, dtype=float).copy()
        self.edge_faces = {}
        self._distance_context = {}
        self.vertex_neighbors = {
            vertex_id: set() for vertex_id in range(len(self.vertices))
        }
        self.vertex_faces = {
            vertex_id: set() for vertex_id in range(len(self.vertices))
        }

        for face_index, face in enumerate(self.faces):
            if not face.visible:
                continue

            vertices = self._visible_face_vertices(face)
            if len(set(vertices)) < 3:
                continue

            for vertex_id in vertices:
                self.vertex_faces[vertex_id].add(face_index)

            for first, second in (
                (vertices[0], vertices[1]),
                (vertices[1], vertices[2]),
                (vertices[2], vertices[0])
            ):
                edge = tuple(sorted((first, second)))
                self.edge_faces.setdefault(edge, set()).add(face_index)
                self.vertex_neighbors[first].add(second)
                self.vertex_neighbors[second].add(first)

        return list(self.edge_faces)

    def collect_edge_neighborhood(self, a, b):
        """[STRUCTURE] Retourne les faces incidentes à a ou b sans cloner le maillage."""
        face_indices = self.vertex_faces.get(a, set()) | self.vertex_faces.get(b, set())
        return [
            self.faces[index].clone()
            for index in sorted(face_indices)
            if self.faces[index].visible
        ]

    # ==================================================================
    # [HOPPE] Calculs d'Énergie originaux (Hoppe SIGGRAPH 1996)
    # E_total = E_dist + lambda_spring * E_spring + lambda_disc * E_disc
    # ==================================================================
    @staticmethod
    def _point_triangle_distance_squared(point, first, second, third):
        """[HOPPE] Calcule la distance au carré entre un point et un triangle."""
        edge_first = second - first
        edge_second = third - first
        offset = point - first
        dot_first = np.dot(edge_first, offset)
        dot_second = np.dot(edge_second, offset)
        if dot_first <= 0 and dot_second <= 0:
            return float(np.dot(offset, offset))

        offset_second = point - second
        dot_edge = np.dot(edge_first, offset_second)
        dot_second_edge = np.dot(edge_second - edge_first, offset_second)
        if dot_edge >= 0 and dot_second_edge <= dot_edge:
            return float(np.dot(offset_second, offset_second))

        offset_third = point - third
        edge_third = first - third
        dot_third = np.dot(edge_third, offset_third)
        dot_second_third = np.dot(edge_second - edge_third, offset_third)
        if dot_third >= 0 and dot_second_third <= dot_third:
            return float(np.dot(offset_third, offset_third))

        denominator = (
            np.dot(edge_first, edge_first)
            * np.dot(edge_second, edge_second)
            - np.dot(edge_first, edge_second) ** 2
        )
        if denominator <= 1e-12:
            return min(
                np.dot(offset, offset),
                np.dot(offset_second, offset_second),
                np.dot(offset_third, offset_third)
            )
        coordinate_first = (
            np.dot(edge_first, edge_second) * dot_second
            - np.dot(edge_second, edge_second) * dot_first
        ) / denominator
        coordinate_second = (
            np.dot(edge_first, edge_second) * dot_first
            - np.dot(edge_first, edge_first) * dot_second
        ) / denominator
        if coordinate_first >= 0 and coordinate_second >= 0:
            if coordinate_first + coordinate_second <= 1:
                projection = first + coordinate_first * edge_first + coordinate_second * edge_second
                difference = point - projection
                return float(np.dot(difference, difference))

        return min(
            ProgressiveMesh._point_segment_distance_squared(point, first, second),
            ProgressiveMesh._point_segment_distance_squared(point, second, third),
            ProgressiveMesh._point_segment_distance_squared(point, third, first)
        )

    @staticmethod
    def _point_segment_distance_squared(point, first, second):
        direction = second - first
        length_squared = np.dot(direction, direction)
        if length_squared <= 1e-12:
            difference = point - first
            return float(np.dot(difference, difference))
        parameter = np.clip(np.dot(point - first, direction) / length_squared, 0.0, 1.0)
        difference = point - (first + parameter * direction)
        return float(np.dot(difference, difference))

    def _distance_energy(self, a, b, position):
        """Approxime Edist par les points originaux des faces affectées."""
        if self.reference_vertices is None:
            self.reference_vertices = np.asarray(self.vertices, dtype=float).copy()
        context = self._distance_context.get((a, b))
        if context is None:
            samples = set()
            triangles = []
            affected = self.vertex_faces.get(a, set()) | self.vertex_faces.get(b, set())
            for face_index in affected:
                face = self.faces[face_index]
                if not face.visible:
                    continue
                original_vertices = self._visible_face_vertices(face)
                vertices = tuple(
                    b if vertex == a else vertex for vertex in original_vertices
                )
                if len(set(vertices)) < 3:
                    continue
                samples.update(original_vertices)
                triangles.append(vertices)
            context = (tuple(samples), tuple(triangles))
            self._distance_context[(a, b)] = context
        samples, triangles = context
        if not triangles:
            return float("inf")
        return sum(
            min(
                self._point_triangle_distance_squared(
                    self.reference_vertices[index],
                    *(
                        position if vertex == b else self.vertices[vertex]
                        for vertex in triangle
                    )
                )
                for triangle in triangles
            )
            for index in samples
        )

    def _spring_energy(self, a, b, position):
        """Calcule l'énergie ressort locale après la contraction."""
        affected_neighbors = (
            self.vertex_neighbors.get(a, set())
            | self.vertex_neighbors.get(b, set())
        )
        energy = 0.0
        counted_edges = set()
        for neighbor in affected_neighbors:
            if neighbor in (a, b):
                continue
            edge = tuple(sorted((b, neighbor)))
            if edge in counted_edges:
                continue
            counted_edges.add(edge)
            energy += float(np.sum((position - self.vertices[neighbor]) ** 2))
        return energy

    def _discontinuity_energy(self, a, b, position):
        """Préserve les arêtes de frontière pendant une contraction.

        Les extrémités et le milieu des arêtes de frontière affectées sont
        projetés sur les nouvelles arêtes de frontière. Une frontière qui
        disparaît reçoit une pénalité infinie.
        """
        if self.reference_vertices is None:
            self.reference_vertices = np.asarray(self.vertices, dtype=float).copy()

        affected_vertices = {a, b}
        affected_vertices.update(self.vertex_neighbors.get(a, set()))
        affected_vertices.update(self.vertex_neighbors.get(b, set()))
        boundary_edges = [
            edge for edge, faces in self.edge_faces.items()
            if len(faces) == 1 and set(edge) & affected_vertices
        ]
        if not boundary_edges:
            return 0.0

        def mapped_position(vertex):
            if vertex == a or vertex == b:
                return position
            return self.vertices[vertex]

        new_boundary_edges = set()
        for edge in boundary_edges:
            mapped_edge = tuple(sorted(
                b if vertex == a else vertex
                for vertex in edge
            ))
            if mapped_edge[0] != mapped_edge[1]:
                new_boundary_edges.add(mapped_edge)

        if not new_boundary_edges:
            return float("inf")

        energy = 0.0
        for first, second in boundary_edges:
            first_point = self.reference_vertices[first]
            second_point = self.reference_vertices[second]
            samples = (
                first_point,
                second_point,
                (first_point + second_point) / 2.0
            )
            for sample in samples:
                energy += min(
                    self._point_segment_distance_squared(
                        sample,
                        mapped_position(edge_first),
                        mapped_position(edge_second)
                    )
                    for edge_first, edge_second in new_boundary_edges
                )
        return float(energy)

    # [HOPPE] Énergie totale de contraction : E = Edist + lambda_spring * Espring + lambda_disc * Edisc
    def _total_collapse_energy(self, a, b, position):
        """[HOPPE] Retourne l'énergie totale de contraction."""
        distance_energy = self._distance_energy(a, b, position)
        spring_energy = self._spring_energy(a, b, position)
        discontinuity_energy = self._discontinuity_energy(a, b, position)
        representation_energy = 1.0
        return (
            distance_energy
            + self.lambda_spring * spring_energy
            + self.lambda_rep * representation_energy
            + self.lambda_disc * discontinuity_energy
        )

    # ==================================================================
    # [HYBRIDE QEM / HOPPE] Sélection de la position optimale du sommet contracté
    # ==================================================================
    def _candidate_position(self, a, b):
        """[HYBRIDE] Calcule la position optimale : QEM (Garland 1997), Chen & Nishita (2002) ou Hoppe (1996)."""
        if self.use_chen:
            if not self._chen_face_normals:
                self._chen_face_normals = chen.compute_face_normals(self.vertices, self.faces)
            res = chen.choose_half_edge_direction(
                a,
                b,
                self.edge_faces,
                self.vertices,
                self.vertex_faces,
                self._chen_face_normals,
                self._chen_unremovable,
            )
            if res is None:
                return np.asarray(self.vertices[b], dtype=float), float("inf")
            _, _, position, cost = res
            return position, cost

        if self.use_qem:
            # Branche QEM (avec hook saillance Limper 2016 facultatif)
            if not self.quadrics10:
                self.compute_initial_quadrics()
            return qem.optimal_candidate_position(
                self.quadrics10, self.vertices, a, b, self.saliency_weight, self.vertex_saliency
            )

        # Branche HOPPE originale (1996) basée sur l'énergie totale (distance + ressort + discontinuité)
        candidates = [
            self.vertices[a],
            self.vertices[b],
            (self.vertices[a] + self.vertices[b]) / 2.0
        ]
        best_position = candidates[0]
        best_cost = float("inf")
        for position in candidates:
            cost = self._total_collapse_energy(a, b, position)
            if np.isfinite(cost) and cost < best_cost:
                best_position = position
                best_cost = cost
        return np.asarray(best_position, dtype=float), max(0.0, best_cost)

    def edge_length(self, a, b):
        """[HOPPE] Retourne l'énergie totale de contraction de l'arête (a, b)."""
        if (a, b) not in self.edge_faces and (b, a) not in self.edge_faces:
            return float("inf")
        _, cost = self._candidate_position(a, b)
        return cost

    # ==================================================================
    # [VALIDITÉ TOPOLOGIQUE & GÉOMÉTRIQUE] Link condition, frontières et normales
    # ==================================================================
    def _link_condition(self, vs, vt):
        """[TOPOLOGIE] Vérifie la condition de lien pour garantir que le maillage reste 2-manifold."""
        edge = tuple(sorted((vs, vt)))
        incident_faces = self.edge_faces.get(edge, set())
        if not incident_faces or len(incident_faces) > 2:
            return False

        common_neighbors = (
            self.vertex_neighbors.get(vs, set())
            & self.vertex_neighbors.get(vt, set())
        )
        opposite_vertices = set()
        for face_index in incident_faces:
            vertices = set(self._visible_face_vertices(self.faces[face_index]))
            vertices.discard(vs)
            vertices.discard(vt)
            if len(vertices) != 1:
                return False
            opposite_vertices.update(vertices)

        return common_neighbors == opposite_vertices

    def _is_boundary_vertex(self, v):
        """[TOPOLOGIE] Vérifie en O(1) si le sommet v appartient à au moins une arête de bord."""
        for nbr in self.vertex_neighbors.get(v, ()):
            edge = (v, nbr) if v < nbr else (nbr, v)
            faces = self.edge_faces.get(edge)
            if faces is not None and len(faces) == 1:
                return True
        return False

    def is_valid_collapse(self, vs, vt, candidate=None):
        """[GÉOMÉTRIE & TOPOLOGIE] Contrôle de bord, manifold, link condition et non-retournement de normales."""
        if vs == vt or not (0 <= vs < len(self.vertices)) or not (0 <= vt < len(self.vertices)):
            return False
        edge = tuple(sorted((vs, vt)))
        if edge not in self.edge_faces:
            return False
        if len(self.edge_faces[edge]) == 1:
            return False
        if self._is_boundary_vertex(vs) != self._is_boundary_vertex(vt):
            return False
        if not self._link_condition(vs, vt):
            return False

        if candidate is None:
            candidate, _ = self._candidate_position(vs, vt)

        cand_x, cand_y, cand_z = float(candidate[0]), float(candidate[1]), float(candidate[2])
        verts = self.vertices
        affected = (
            self.vertex_faces.get(vs, set())
            | self.vertex_faces.get(vt, set())
        )
        for face_index in affected:
            face = self.faces[face_index]
            if not face.visible:
                continue

            i0, i1, i2 = face.a, face.b, face.c
            n0 = vt if i0 == vs else i0
            n1 = vt if i1 == vs else i1
            n2 = vt if i2 == vs else i2
            if n0 == n1 or n1 == n2 or n2 == n0:
                continue

            p0, p1, p2 = verts[i0], verts[i1], verts[i2]
            ux, uy, uz = float(p1[0] - p0[0]), float(p1[1] - p0[1]), float(p1[2] - p0[2])
            wx, wy, wz = float(p2[0] - p0[0]), float(p2[1] - p0[1]), float(p2[2] - p0[2])
            ox = uy * wz - uz * wy
            oy = uz * wx - ux * wz
            oz = ux * wy - uy * wx
            old_norm_sq = ox * ox + oy * oy + oz * oz

            q0x, q0y, q0z = (cand_x, cand_y, cand_z) if n0 == vt else (float(verts[n0][0]), float(verts[n0][1]), float(verts[n0][2]))
            q1x, q1y, q1z = (cand_x, cand_y, cand_z) if n1 == vt else (float(verts[n1][0]), float(verts[n1][1]), float(verts[n1][2]))
            q2x, q2y, q2z = (cand_x, cand_y, cand_z) if n2 == vt else (float(verts[n2][0]), float(verts[n2][1]), float(verts[n2][2]))

            nux, nuy, nuz = q1x - q0x, q1y - q0y, q1z - q0z
            nwx, nwy, nwz = q2x - q0x, q2y - q0y, q2z - q0z
            nx = nuy * nwz - nuz * nwy
            ny = nuz * nwx - nux * nwz
            nz = nux * nwy - nuy * nwx
            new_norm_sq = nx * nx + ny * ny + nz * nz

            if new_norm_sq <= 1e-24:
                return False
            if old_norm_sq > 1e-24 and (ox * nx + oy * ny + oz * nz) <= 0.0:
                return False

        return True

    # ==================================================================
    # [FILE DE PRIORITÉ & CONTRACTION PROGRESSIVE] Gestion du tas et collapses
    # ==================================================================
    def _push_edge(self, edge):
        """[FILE] Calcule la priorité de l'arête (QEM, Chen ou Hoppe) et l'insère dans le tas."""
        ver = self._edge_versions.get(edge, 0) + 1
        self._edge_versions[edge] = ver
        if self.use_chen:
            res = chen.choose_half_edge_direction(
                edge[0],
                edge[1],
                self.edge_faces,
                self.vertices,
                self.vertex_faces,
                self._chen_face_normals,
                self._chen_unremovable,
            )
            if res is None:
                return
            vs, vt, candidate, cost = res
            heapq.heappush(self._heap, (cost, edge, ver, (vs, vt, candidate)))
        elif self.use_qem:
            # [QEM] Priorité basée sur l'erreur quadrique v^T Q v (+ saillance facultative)
            candidate, cost = self._candidate_position(*edge)
            heapq.heappush(self._heap, (cost, edge, ver, candidate))
        else:
            # [HOPPE] Priorité basée sur la longueur/énergie de l'arête
            first, second = self.vertices[edge[0]], self.vertices[edge[1]]
            diff = first - second
            priority = float(np.dot(diff, diff))
            heapq.heappush(self._heap, (priority, edge, ver, None))

    def _build_priority_queue(self):
        """[FILE] Construit la file de priorité initiale."""
        self._heap = []
        self._edge_versions = {}
        self._pending_candidate = None
        self._pending_edge = None
        if self.use_qem and not self.quadrics10:
            self.compute_initial_quadrics()
        if self.use_chen and not self._chen_face_normals:
            self._chen_face_normals = chen.compute_face_normals(self.vertices, self.faces)
        edges = list(self.edge_faces) if (self.use_chen and self.edge_faces) else self.collect_edges()
        for edge in edges:
            self._push_edge(edge)

    def shortest_edge(self):
        """[FILE] Extrait l'arête valide de coût minimal (tas-min)."""
        if not self._heap:
            self._build_priority_queue()
        while self._heap:
            cost, edge, version, cached_cand = heapq.heappop(self._heap)
            if version != self._edge_versions.get(edge):
                continue
            if edge not in self.edge_faces:
                continue
            if self.use_chen:
                if cached_cand is not None:
                    vs, vt, candidate = cached_cand
                else:
                    res = chen.choose_half_edge_direction(
                        edge[0],
                        edge[1],
                        self.edge_faces,
                        self.vertices,
                        self.vertex_faces,
                        self._chen_face_normals,
                        self._chen_unremovable,
                    )
                    if res is None:
                        continue
                    vs, vt, candidate, cost = res
            else:
                vs, vt = edge
                if cached_cand is not None:
                    candidate = cached_cand
                else:
                    candidate, cost = self._candidate_position(*edge)
            if not np.isfinite(cost):
                continue
            if self.is_valid_collapse(vs, vt, candidate=candidate):
                self._pending_edge = edge
                self._pending_candidate = (candidate, cost)
                return (vs, vt)
        return None

    def collapse_edge(self, vs, vt):
        """[STRUCTURE] Contracte vs vers vt et met à jour localement la topologie en O(1)."""
        edge = (vs, vt) if vs < vt else (vt, vs)
        if self._pending_edge == edge and self._pending_candidate is not None:
            candidate, cost = self._pending_candidate
        else:
            candidate, cost = self._candidate_position(vs, vt)
        if not self.is_valid_collapse(vs, vt, candidate=candidate):
            self._pending_edge = None
            self._pending_candidate = None
            return False

        removed_faces = []
        modified_faces = []
        target_position = np.copy(self.vertices[vt])

        # Voisins affectés pour mise à jour locale de vertex_neighbors
        affected_nbrs = {vs, vt}
        affected_nbrs.update(self.vertex_neighbors.get(vs, ()))
        affected_nbrs.update(self.vertex_neighbors.get(vt, ()))

        affected = sorted(
            self.vertex_faces.get(vs, set())
            | self.vertex_faces.get(vt, set())
        )

        for face_index in affected:
            face = self.faces[face_index]
            if not face.visible:
                continue

            old_face = face.clone()
            old_verts = self._visible_face_vertices(face)
            new_vertices = [
                vt if vertex == vs else vertex
                for vertex in old_verts
            ]
            if len(set(new_vertices)) < 3:
                # 1. Face dégénérée : elle est supprimée
                removed_faces.append((face_index, old_face))
                face.visible = False
                self.deleted_faces.add(face_index)
                if self._visible_face_count is not None:
                    self._visible_face_count -= 1

                # Détacher cette face de ses sommets et arêtes
                for v in old_verts:
                    if v in self.vertex_faces:
                        self.vertex_faces[v].discard(face_index)
                for i in range(3):
                    u, w = old_verts[i], old_verts[(i + 1) % 3]
                    e = (u, w) if u < w else (w, u)
                    ef = self.edge_faces.get(e)
                    if ef is not None:
                        ef.discard(face_index)
                        if not ef:
                            del self.edge_faces[e]
                continue

            # 2. Face modifiée : vs est remplacé par vt
            face.a, face.b, face.c = new_vertices
            modified_faces.append((face_index, old_face, face.clone()))

            # Mettre à jour l'incidence sommet-face
            if vs in self.vertex_faces:
                self.vertex_faces[vs].discard(face_index)
            self.vertex_faces.setdefault(vt, set()).add(face_index)

            # Mettre à jour les arêtes qui contenaient vs
            for i in range(3):
                u_old, w_old = old_verts[i], old_verts[(i + 1) % 3]
                u_new, w_new = new_vertices[i], new_vertices[(i + 1) % 3]
                if (u_old, w_old) != (u_new, w_new):
                    e_old = (u_old, w_old) if u_old < w_old else (w_old, u_old)
                    ef = self.edge_faces.get(e_old)
                    if ef is not None:
                        ef.discard(face_index)
                        if not ef:
                            del self.edge_faces[e_old]
                    e_new = (u_new, w_new) if u_new < w_new else (w_new, u_new)
                    self.edge_faces.setdefault(e_new, set()).add(face_index)

        # Mettre à jour vertex_neighbors localement pour les sommets affectés
        self.vertex_neighbors[vs] = set()
        for u in affected_nbrs:
            if u == vs:
                continue
            nbrs = set()
            for fid in self.vertex_faces.get(u, ()):
                f = self.faces[fid]
                for v in (f.a, f.b, f.c):
                    if v != u:
                        nbrs.add(v)
            self.vertex_neighbors[u] = nbrs

        self.vertices[vt] = candidate
        if self.use_qem and self.quadrics10:
            qem.update_quadrics_on_collapse(self.quadrics10, vs, vt)
        if self.use_chen:
            for fid, _ in removed_faces:
                self._chen_face_normals.pop(fid, None)
            for fid, _, _ in modified_faces:
                f = self.faces[fid]
                p0, p1, p2 = self.vertices[f.a], self.vertices[f.b], self.vertices[f.c]
                ux, uy, uz = float(p1[0] - p0[0]), float(p1[1] - p0[1]), float(p1[2] - p0[2])
                vx, vy, vz = float(p2[0] - p0[0]), float(p2[1] - p0[1]), float(p2[2] - p0[2])
                nx = uy * vz - uz * vy
                ny = uz * vx - ux * vz
                nz = ux * vy - uy * vx
                norm = math.sqrt(nx * nx + ny * ny + nz * nz)
                if norm > 1e-12:
                    inv = 1.0 / norm
                    self._chen_face_normals[fid] = (nx * inv, ny * inv, nz * inv)
                else:
                    self._chen_face_normals.pop(fid, None)

        self.history.append(
            Collapse(
                vs,
                self.vertices[vs],
                vt,
                target_position,
                removed_faces,
                modified_faces,
                cost
            )
        )

        if self._heap:
            for neighbor in self.vertex_neighbors.get(vt, ()):
                nbr_edge = (vt, neighbor) if vt < neighbor else (neighbor, vt)
                if nbr_edge in self.edge_faces:
                    self._push_edge(nbr_edge)

        self._pending_edge = None
        self._pending_candidate = None
        return True

    # Progressive mesh et vertex split
    def simplify(self, target_ratio=0.1):
        if not 0 < target_ratio <= 1:
            raise ValueError("target_ratio doit être compris entre 0 et 1.")

        self.collect_edges()
        if self.use_qem and not self.quadrics10:
            self.compute_initial_quadrics()

        # Hook automatique pour la saillance (Limper 2016)
        if self.saliency_weight > 0.0 and self.vertex_saliency is None:
            print("    - Calcul de la saillance de surface (Limper 2016)...")
            self.vertex_saliency = saillance.calculer_saillance(self)

        initial_faces = self.count_visible_faces()
        target_faces = max(1, int(initial_faces * target_ratio))

        if self.use_chen:
            self._chen_face_normals = chen.compute_face_normals(self.vertices, self.faces)
            levels = (
                (0.15, True),
                (0.40, True),
                (0.85, True),
                (1.40, False),
                (2.10, False),
            )
            for level_threshold, lock_sharp in levels:
                if self.count_visible_faces() <= target_faces:
                    break
                self._chen_unremovable = chen.classify_unremovable_vertices(
                    self.edge_faces,
                    self.vertices,
                    self._chen_face_normals,
                    level_threshold=level_threshold,
                    lock_sharp_edges=lock_sharp,
                )
                self._build_priority_queue()
                while self.count_visible_faces() > target_faces:
                    edge = self.shortest_edge()
                    if edge is None or not self.collapse_edge(*edge):
                        break
            return

        self._build_priority_queue()

        while self.count_visible_faces() > target_faces:
            edge = self.shortest_edge()
            if edge is None or not self.collapse_edge(*edge):
                break

    def vertex_split(self, collapse=None):
        """Annule le dernier collapse et restaure le niveau de détail précédent."""
        if collapse is None:
            if not self.history:
                return False
            collapse = self.history[-1]
        elif not self.history or collapse is not self.history[-1]:
            raise ValueError("Les vertex split doivent suivre l'ordre inverse.")

        self.vertices[collapse.target_vertex_id] = np.copy(
            collapse.target_vertex_position
        )
        for face_index, old_face, _ in collapse.modified_faces:
            self.faces[face_index].copy(old_face)
        for face_index, old_face in collapse.removed_faces:
            self.faces[face_index].copy(old_face)
            self.faces[face_index].visible = True
            self.deleted_faces.discard(face_index)
            if self._visible_face_count is not None:
                self._visible_face_count += 1

        self.history.pop()
        self.collect_edges()
        return True

    def count_visible_faces(self):
        if self._visible_face_count is None:
            self._visible_face_count = sum(face.visible for face in self.faces)
        return self._visible_face_count

    def get_level(self, number_of_splits):
        """Retourne le niveau obtenu après un nombre de vertex splits."""
        if number_of_splits < 0:
            raise ValueError("number_of_splits doit être positif.")
        for _ in range(min(number_of_splits, len(self.history))):
            self.vertex_split()
        return self

    # ==================================================================
    # [EXPORT OBJA] Écriture progressive (Maillage base M^0 + vertex splits)
    # ==================================================================
    @staticmethod
    def _clean_coords(position, decimals=6):
        """[EXPORT OBJA] Formate proprement les coordonnées flottantes pour éviter le bruit sur 18 décimales."""
        return [round(float(c), decimals) for c in position]

    
    def write_obja(self, output, decimals=6, bits=16):
        self._validate_bits(bits)

        output_model = obja.Output(output, random_color=False)

        vertex_size = (3 * bits + 7) // 8
        face_size = 7
        edit_face_size = 5

        cumulative_size = 0
        emitted_positions = {}
        emitted_faces = set()

        ev_written = 0
        ev_skipped = 0

        def write_size():
            print(f"s {cumulative_size}", file=output)

        def ensure_vertex(vertex_id, position=None):
            nonlocal cumulative_size

            if vertex_id not in output_model.vertex_mapping:
                if position is None:
                    position = self.vertices[vertex_id]

                clean_pos = self._clean_coords(position, decimals)
                output_model.add_vertex(vertex_id, clean_pos)
                emitted_positions[vertex_id] = clean_pos
                cumulative_size += vertex_size

        def ensure_face_vertices(face):
            ensure_vertex(face.a)
            ensure_vertex(face.b)
            ensure_vertex(face.c)

        used_vertices = set()

        for face in self.faces:
            if face.visible:
                used_vertices.update(self._visible_face_vertices(face))

        for vertex_id in sorted(used_vertices):
            ensure_vertex(vertex_id)

        for face_index, face in enumerate(self.faces):
            if face.visible:
                output_model.add_face(face_index, face)
                emitted_faces.add(face_index)
                cumulative_size += face_size

        write_size()

        for collapse in reversed(self.history):
            ensure_vertex(
                collapse.vertex_id,
                collapse.vertex_position
            )

            target_clean = self._clean_coords(
                collapse.target_vertex_position,
                decimals
            )

            if collapse.target_vertex_id not in output_model.vertex_mapping:
                ensure_vertex(
                    collapse.target_vertex_id,
                    target_clean
                )

            elif emitted_positions.get(collapse.target_vertex_id) != target_clean:
                output_model.edit_vertex(
                    collapse.target_vertex_id,
                    target_clean
                )

                emitted_positions[collapse.target_vertex_id] = target_clean
                cumulative_size += vertex_size
                ev_written += 1

            else:
                ev_skipped += 1

            for face_index, old_face, new_face in collapse.modified_faces:
                if face_index in emitted_faces:
                    ensure_face_vertices(old_face)

                    output_model.edit_face(face_index, old_face)
                    cumulative_size += edit_face_size

            for face_index, old_face in collapse.removed_faces:
                ensure_face_vertices(old_face)

                output_model.add_face(face_index, old_face)
                emitted_faces.add(face_index)
                cumulative_size += face_size

            write_size()

        self._last_obja_stats = {
            "ev_written": ev_written,
            "ev_skipped": ev_skipped,
            "estimated_size": cumulative_size
        }


    # ==================================================================
    # [QUANTIFICATION] Encodage et réduction du nombre de bits des coordonnées 3D
    # ==================================================================
    @staticmethod
    def _validate_bits(bits):
        """[QUANTIFICATION] Valide que le nombre de bits est compris entre 1 et 32."""
        if not isinstance(bits, int) or not 1 <= bits <= 32:
            raise ValueError("bits doit être un entier compris entre 1 et 32.")

    @staticmethod
    def _quantize_positions(positions, bits):
        """[QUANTIFICATION] Projette un ensemble de positions 3D sur une grille discrète à (2^bits - 1) niveaux."""
        ProgressiveMesh._validate_bits(bits)
        values = np.asarray(positions, dtype=float)
        if values.size == 0:
            return [], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        if not np.all(np.isfinite(values)):
            raise ValueError("Les positions doivent être finies.")
        minimum = np.min(values, axis=0)
        maximum = np.max(values, axis=0)
        extent = np.where(maximum - minimum == 0, 1.0, maximum - minimum)
        scale = (1 << bits) - 1
        quantized = np.rint((values - minimum) / extent * scale)
        return quantized.astype(int).tolist(), minimum.tolist(), maximum.tolist()

    @staticmethod
    def dequantize_vertices(data, bounds, bits=16):
        """[QUANTIFICATION] Reconstruit des positions flottantes à partir de positions quantifiées."""
        ProgressiveMesh._validate_bits(bits)
        minimum, maximum = (
            np.asarray(value, dtype=float) for value in bounds
        )
        extent = np.where(maximum - minimum == 0, 1.0, maximum - minimum)
        scale = float((1 << bits) - 1)
        return minimum + np.asarray(data, dtype=float) / scale * extent

    def quantize_vertices(self, bits=16):
        """[QUANTIFICATION] Quantifie puis déquantifie les sommets du maillage courant."""
        quantized, minimum, maximum = self._quantize_positions(
            self.vertices,
            bits
        )
        self.vertices = [
            value for value in self.dequantize_vertices(
                quantized,
                (minimum, maximum),
                bits
            )
        ]
        return quantized, (minimum, maximum)

    def quantize_history(self, bits=16):
        """[QUANTIFICATION] Retourne les positions quantifiées de l'historique."""
        positions = []
        for collapse in self.history:
            positions.extend([
                collapse.vertex_position,
                collapse.target_vertex_position
            ])
        values, minimum, maximum = self._quantize_positions(positions, bits)
        return {
            "values": values,
            "minimum": minimum,
            "maximum": maximum,
            "bits": bits
        }

    def quantize_model(self, bits=16, decimals=6):
        """
        [QUANTIFICATION] Quantifie puis déquantifie sur une boîte englobante globale l'ensemble des
        positions du maillage de base (self.vertices) ET de l'historique (self.history)
        pour que l'export .obja écrive directement les coordonnées quantifiées.
        """
        self._validate_bits(bits)
        num_v = len(self.vertices)
        all_positions = [np.copy(v) for v in self.vertices]
        for collapse in self.history:
            all_positions.append(np.copy(collapse.vertex_position))
            all_positions.append(np.copy(collapse.target_vertex_position))

        if not all_positions:
            return {"mean": 0.0, "maximum": 0.0, "bits": bits}

        original_array = np.asarray(all_positions, dtype=float)
        quantized, minimum, maximum = self._quantize_positions(original_array, bits)
        dequantized = self.dequantize_vertices(quantized, (minimum, maximum), bits)
        if decimals is not None:
            dequantized = np.round(dequantized, decimals)

        error_stats = self.measure_quantization_error(original_array, dequantized)
        error_stats["bits"] = bits

        # Mise à jour directe des sommets du maillage de base M^0
        for i in range(num_v):
            self.vertices[i] = np.copy(dequantized[i])

        # Mise à jour directe des positions de chaque vertex split dans l'historique
        offset = num_v
        for collapse in self.history:
            collapse.vertex_position = np.copy(dequantized[offset])
            collapse.target_vertex_position = np.copy(dequantized[offset + 1])
            offset += 2

        return error_stats

    @staticmethod
    def compression_ratio(original_size, compressed_size):
        if original_size <= 0 or compressed_size < 0:
            raise ValueError("Les tailles doivent être positives.")
        return compressed_size / original_size

    @staticmethod
    def measure_quantization_error(original, reconstructed):
        original = np.asarray(original, dtype=float)
        reconstructed = np.asarray(reconstructed, dtype=float)
        if original.shape != reconstructed.shape:
            raise ValueError("Les positions doivent avoir la même forme.")
        errors = np.linalg.norm(original - reconstructed, axis=1)
        return {
            "mean": float(np.mean(errors)) if len(errors) else 0.0,
            "maximum": float(np.max(errors)) if len(errors) else 0.0
        }


# ======================================================================
# [EXÉCUTION & CLI MULTI-MODES] Orchestration (Hoppe, QEM, Saillance, Quantifié)
# ======================================================================
def run_simplification(
    model_path,
    ratio=0.1,
    mode="qem",
    method=None,
    boundary_weight=10.0,
    saliency_weight=0.0,
    quantize=False,
    bits=16,
    decimals=6,
    output_obja=None,
    stats_out=None
):
    """[EXÉCUTION] Exécute une simplification progressive selon le mode choisi."""
    # Détermination de la catégorie et des paramètres selon le mode
    if mode == "hoppe":
        category = "hoppe"
        use_method = "hoppe"
        sal_weight = 0.0
        do_quantize = False
    elif mode == "saillance":
        category = "saillance"
        use_method = "qem"
        sal_weight = saliency_weight if saliency_weight > 0.0 else 5.0
        do_quantize = False
    elif mode == "quantifie":
        category = "quantifie"
        use_method = "qem"
        sal_weight = 0.0
        do_quantize = True
    elif mode == "chen":
        category = "chen"
        use_method = "chen"
        sal_weight = 0.0
        do_quantize = False
    elif mode == "qem":
        category = "qem"
        use_method = "qem"
        sal_weight = 0.0
        do_quantize = False
    else:
        # Déduction automatique si --mode n'est pas spécifié
        use_method = method if method else "qem"
        if use_method == "hoppe":
            category = "hoppe"
            sal_weight = 0.0
            do_quantize = False
        elif use_method == "chen":
            category = "chen"
            sal_weight = 0.0
            do_quantize = False
        elif saliency_weight > 0.0:
            category = "saillance"
            sal_weight = saliency_weight
            do_quantize = False
        elif quantize:
            category = "quantifie"
            sal_weight = 0.0
            do_quantize = True
        else:
            category = "qem"
            sal_weight = 0.0
            do_quantize = False

    print("=" * 70)
    print(f"  PROGRESSIVE MESH — MODE [{category.upper()}]")
    print("=" * 70)
    print(f"Modèle d'entrée    : {model_path.name}")
    print(f"Méthode choisie    : {use_method.upper()}")
    print(f"Catégorie sortie   : example/{category}/")
    print(f"Ratio cible        : {ratio:.1%}")
    if sal_weight > 0.0:
        print(f"Option Saillance   : ACTIVÉE (poids = {sal_weight:g})")
    if do_quantize:
        print(f"Option Quantifié   : ACTIVÉE ({bits} bits)")

    # 1. Chargement du modèle
    model = ProgressiveMesh(
        use_qem=(use_method == "qem"),
        use_chen=(use_method == "chen"),
        boundary_weight=boundary_weight,
        saliency_weight=sal_weight,
    )
    t0_load = perf_counter()
    model.parse_file(str(model_path))
    t_load = perf_counter() - t0_load

    initial_vertices = len(model.vertices)
    initial_faces = model.count_visible_faces()
    print(f"\n[1] Maillage initial chargé en {t_load:.4f}s :")
    print(f"    - Sommets : {initial_vertices}")
    print(f"    - Faces   : {initial_faces}")

    # 2. Simplification progressive
    print(f"\n[2] Simplification progressive en cours ({category.upper()})...")
    t0_sim = perf_counter()
    model.simplify(target_ratio=ratio)
    t_sim = perf_counter() - t0_sim

    final_faces = model.count_visible_faces()
    num_collapses = len(model.history)
    rate = num_collapses / max(1e-6, t_sim)
    print(f"    - Terminé en {t_sim:.4f}s ({rate:.0f} collapses/sec) !")
    print(f"    - Faces finales : {initial_faces} -> {final_faces} ({final_faces / initial_faces:.1%})")
    print(f"    - Vertex splits enregistrés : {num_collapses}")

    # 3. Quantification géométrique (si activée pour le mode quantifie)
    t_quant = 0.0
    if do_quantize and bits > 0:
        t0_quant = perf_counter()
        q_stats = model.quantize_model(bits=bits, decimals=decimals)
        t_quant = perf_counter() - t0_quant
        print(f"\n[3] Quantification géométrique appliquée ({bits} bits) en {t_quant:.4f}s :")
        print(f"    - Erreur moyenne de quantification  : {q_stats['mean']:.6e}")
        print(f"    - Erreur maximale de quantification : {q_stats['maximum']:.6e}")
    else:
        print(f"\n[3] Coordonnées directes (mode {category}).")

    # 4. Export OBJA progressif : le nom du fichier ne contient QUE le nom et le ratio (sans '_')
    clean_stem = model_path.stem.replace("_", "")
    filename = f"{clean_stem}{ratio:g}.obja"
    if output_obja:
        obja_path = Path(output_obja)
    else:
        obja_path = Path("example") / category / filename
    obja_path.parent.mkdir(parents=True, exist_ok=True)

    export_bits = bits if do_quantize else 32
    t0_export = perf_counter()
    with open(obja_path, "w", encoding="utf-8") as file:
        model.write_obja(file, decimals=decimals, bits=export_bits)
    t_export = perf_counter() - t0_export
    t_comp = t_sim + t_quant + t_export

    obja_bytes = obja_path.stat().st_size
    ev_stats = getattr(model, "_last_obja_stats", {"ev_written": 0, "ev_skipped": 0, "estimated_size": 0})
    print(f"\n[4] Export OBJA progressif (en {t_export:.4f}s) :")
    print(f"    - Chemin du fichier             : {obja_path}")
    print(f"    - Nom du fichier (nom+ratio)    : {obja_path.name}")
    print(f"    - Taille du fichier OBJA        : {obja_bytes:,} octets ({obja_bytes / 1024:.1f} Ko)")
    if ev_stats.get("estimated_size"):
        print(f"    - Taille binaire estimée ({export_bits}b) : {ev_stats['estimated_size']:,} octets ({ev_stats['estimated_size'] / 1024:.1f} Ko)")
    print(f"    - Instructions 'ev' écrites     : {ev_stats['ev_written']} ({ev_stats['ev_skipped']} 'ev' redondants évités)")

    # 5. Métriques de compression (temps + erreur géométrique M^0 vs original)
    comp = None
    try:
        comp = comparaison.compare_files(str(model_path), str(obja_path))
        print(f"\n[5] Métriques de compression et d'erreur géométrique M^0 vs original :")
        print(f"    - Temps de simplification        : {t_sim:.4f}s ({rate:.0f} collapses/sec)")
        print(f"    - Temps total de compression     : {t_comp:.4f}s (sim={t_sim:.4f}s, quant={t_quant:.4f}s, export={t_export:.4f}s)")
        print(f"    - Sommets restants               : {comp['original_vertices']} -> {comp['final_vertices']} (-{comp['vertex_reduction_percent']:.1f}%)")
        print(f"    - Faces restantes                : {comp['original_faces']} -> {comp['final_faces']} (-{comp['face_reduction_percent']:.1f}%)")
        print(f"    - Erreur géométrique moyenne     : {comp['mean_geometric_error']:.6f}")
        print(f"    - Erreur géométrique maximale    : {comp['maximum_geometric_error']:.6f} (Hausdorff)")
        print(f"    - Erreur moyenne normalisée      : {comp['normalized_mean_error_percent']:.4f}% de la diagonale")
        print(f"    - Erreur maximale normalisée     : {comp['normalized_maximum_error_percent']:.4f}% de la diagonale")
    except Exception as e:
        print(f"\n[5] Temps total de compression : {t_comp:.4f}s | Erreur lors du calcul comparatif : {e}")

    if stats_out is not None:
        stats_out.update({
            "mode": category,
            "model": model_path.name,
            "t_load": t_load,
            "t_sim": t_sim,
            "t_quant": t_quant,
            "t_export": t_export,
            "t_comp": t_comp,
            "rate": rate,
            "obja_bytes": obja_bytes,
            "export_bits": export_bits,
            "estimated_size": ev_stats.get("estimated_size", 0),
            "initial_vertices": initial_vertices,
            "initial_faces": initial_faces,
            "final_faces": final_faces,
            "comp": comp,
            "obja_path": obja_path,
        })

    # 6. Validation de la réversibilité
    split_ok = model.vertex_split() if num_collapses > 0 else True
    print(f"\n[6] Test de réversibilité (vertex_split) : {'SUCCÈS' if split_ok else 'ÉCHEC'}")
    print("=" * 70)
    return obja_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Progressive Mesh : Hoppe (1996), QEM (Garland 1997), Saillance (Limper 2016), Quantifié."
    )
    parser.add_argument("model", nargs="?", default="example/bunny.obj", help="Chemin vers le modèle OBJ (défaut: example/bunny.obj)")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio de faces cibles (ex: 0.1 pour 10%%)")
    parser.add_argument(
        "--mode",
        choices=["qem", "hoppe", "saillance", "quantifie", "chen", "all"],
        default=None,
        help="Mode de sortie: 'hoppe', 'qem', 'saillance', 'quantifie', 'chen' ou 'all' (génère les 5)"
    )
    parser.add_argument("--method", choices=["qem", "hoppe", "chen"], default=None, help="Méthode de coût d'erreur (si mode non spécifié)")
    parser.add_argument("--boundary-weight", type=float, default=10.0, help="Poids des plans de contrainte de bord (Garland §6)")
    parser.add_argument("--saliency-weight", type=float, default=0.0, help="Poids de la saillance (défaut: 0.0)")
    parser.add_argument("--quantize", action="store_true", help="Active l'option de quantification géométrique")
    parser.add_argument("--bits", type=int, default=16, help="Nombre de bits de quantification (défaut: 16)")
    parser.add_argument("--decimals", type=int, default=6, help="Nombre de décimales écrites dans l'OBJA (défaut: 6)")
    parser.add_argument("--output-obja", type=str, default=None, help="Chemin de sortie OBJA personnalisé")
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_file():
        alt = Path("example") / model_path.name
        if not alt.suffix:
            alt = alt.with_suffix(".obj")
        if alt.is_file():
            model_path = alt
        else:
            raise FileNotFoundError(f"Impossible de trouver le modèle : {args.model}")

    if args.mode == "all":
        modes = ["qem", "quantifie", "saillance", "hoppe", "chen"]
        print(f"\n>>> GÉNÉRATION DES 4 MODES POUR {model_path.name} (Ratio {args.ratio:.1%}) <<<\n")
        outputs = []
        all_stats = []
        for m in modes:
            st = {}
            out = run_simplification(
                model_path=model_path,
                ratio=args.ratio,
                mode=m,
                boundary_weight=args.boundary_weight,
                saliency_weight=args.saliency_weight,
                bits=args.bits,
                decimals=args.decimals,
                stats_out=st
            )
            outputs.append(out)
            all_stats.append(st)
        print("\n" + "=" * 102)
        print(f"  TABLEAU COMPARATIF — {model_path.name} (Ratio {args.ratio:.1%})")
        print("=" * 102)
        print(f"{'Mode':<11} | {'Temps (s)':>9} | {'Vitesse':>11} | {'Hausdorff':>10} | {'Err Moy (%)':>11} | {'Taille (Ko)':>11} | {'Est.Bin(Ko)':>13}")
        print("-" * 102)
        for st in all_stats:
            comp = st.get("comp") or {}
            haus = comp.get("maximum_geometric_error", 0.0)
            nmean = comp.get("normalized_mean_error_percent", 0.0)
            size_kb = st.get("obja_bytes", 0) / 1024.0
            est_kb = f"{st.get('estimated_size', 0) / 1024.0:.1f} ({st.get('export_bits', 32)}b)"
            speed = f"{st.get('rate', 0.0):.0f} c/s"
            print(f"{st.get('mode', ''):<11} | {st.get('t_comp', 0.0):>9.4f} | {speed:>11} | {haus:>10.6f} | {nmean:>10.4f}% | {size_kb:>11.1f} | {est_kb:>13}")
        print("=" * 102)
        print("\n>>> TOUS LES FICHIERS ONT ÉTÉ GÉNÉRÉS AVEC SUCCÈS :")
        for out in outputs:
            print(f"    - {out}")
    else:
        run_simplification(
            model_path=model_path,
            ratio=args.ratio,
            mode=args.mode,
            method=args.method,
            boundary_weight=args.boundary_weight,
            saliency_weight=args.saliency_weight,
            quantize=args.quantize,
            bits=args.bits,
            decimals=args.decimals,
            output_obja=args.output_obja
        )


if __name__ == "__main__":
    main()
