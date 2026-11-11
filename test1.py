#!/usr/bin/env python3

import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct
import sys
from time import perf_counter
import zlib

import numpy as np

import obja
import comparaison
import saillance


# ----------------------------------------------------------------------
# Quadriques d'Erreur (Garland & Heckbert, SIGGRAPH 1997)
# Représentation compacte symétrique 4x4 en 10 flottants
# ----------------------------------------------------------------------
def plane_to_quadric10(a: float, b: float, c: float, d: float, weight: float = 1.0) -> list[float]:
    """Construit les 10 coefficients symétriques de la quadrique fondamentale weight * p p^T."""
    return [
        weight * a * a,
        weight * a * b,
        weight * a * c,
        weight * a * d,
        weight * b * b,
        weight * b * c,
        weight * b * d,
        weight * c * c,
        weight * c * d,
        weight * d * d,
    ]


def eval_quadric10(q: list[float], x: float, y: float, z: float) -> float:
    """Évalue v^T Q v via la formule développée de Garland (1997, §4 note 2)."""
    val = (
        q[0] * x * x
        + 2.0 * q[1] * x * y
        + 2.0 * q[2] * x * z
        + 2.0 * q[3] * x
        + q[4] * y * y
        + 2.0 * q[5] * y * z
        + 2.0 * q[6] * y
        + q[7] * z * z
        + 2.0 * q[8] * z
        + q[9]
    )
    return val if val > 0.0 else 0.0


def add_quadric10_inplace(dst: list[float], src: list[float]) -> None:
    """Accumule dst += src sur les 10 coefficients symétriques."""
    for i in range(10):
        dst[i] += src[i]


def sum_quadric10(q1: list[float], q2: list[float]) -> list[float]:
    """Retourne q1 + q2 sous forme de liste de 10 flottants."""
    return [q1[i] + q2[i] for i in range(10)]


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
        self.use_qem = bool(use_qem)
        self.boundary_weight = float(boundary_weight)
        self.saliency_weight = float(saliency_weight)
        self.vertex_saliency = None  # Point d'ancrage pour le collègue (Saillance LCE)
        self.quadrics10 = []
        self._heap = []
        self._heap_version = 0
        self._distance_context = {}
        self._pending_candidate = None
        self._pending_edge = None
        self._visible_face_count = None

    # ------------------------------------------------------------------
    # Structures de voisinage et Quadrics (Garland & Heckbert 1997)
    # ------------------------------------------------------------------
    def compute_initial_quadrics(self):
        """Initialise les quadriques 10-float par sommet selon Garland & Heckbert (1997)."""
        num_v = len(self.vertices)
        self.quadrics10 = [[0.0] * 10 for _ in range(num_v)]

        # 1. Quadriques des faces (pondérées par l'aire du triangle)
        for face in self.faces:
            if not face.visible:
                continue
            p0 = self.vertices[face.a]
            p1 = self.vertices[face.b]
            p2 = self.vertices[face.c]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-12:
                continue
            normal /= norm
            area = 0.5 * norm
            d = -float(np.dot(normal, p0))
            kq = plane_to_quadric10(normal[0], normal[1], normal[2], d, weight=area)
            add_quadric10_inplace(self.quadrics10[face.a], kq)
            add_quadric10_inplace(self.quadrics10[face.b], kq)
            add_quadric10_inplace(self.quadrics10[face.c], kq)

        # 2. Plans de contrainte perpendiculaires sur les bords (Garland §6)
        if self.boundary_weight > 0.0:
            for edge, faces in self.edge_faces.items():
                if len(faces) == 1:
                    u, w = edge
                    pu, pw = self.vertices[u], self.vertices[w]
                    edge_vec = pw - pu
                    edge_len = float(np.linalg.norm(edge_vec))
                    if edge_len <= 1e-12:
                        continue
                    face_idx = next(iter(faces))
                    f = self.faces[face_idx]
                    fn = np.cross(self.vertices[f.b] - self.vertices[f.a], self.vertices[f.c] - self.vertices[f.a])
                    fn_norm = float(np.linalg.norm(fn))
                    if fn_norm <= 1e-12:
                        continue
                    fn /= fn_norm
                    # Plan perpendiculaire à la face passant par l'arête de bord
                    perp = np.cross(edge_vec, fn)
                    perp_norm = float(np.linalg.norm(perp))
                    if perp_norm <= 1e-12:
                        continue
                    perp /= perp_norm
                    d_perp = -float(np.dot(perp, pu))
                    kq_bnd = plane_to_quadric10(perp[0], perp[1], perp[2], d_perp, weight=self.boundary_weight * edge_len)
                    add_quadric10_inplace(self.quadrics10[u], kq_bnd)
                    add_quadric10_inplace(self.quadrics10[w], kq_bnd)

    def qem_cost(self, a, b, position):
        """Évalue l'erreur quadrique QEM (Garland 1997) + hook saillance pour le collègue."""
        if not self.quadrics10 or len(self.quadrics10) != len(self.vertices):
            self.compute_initial_quadrics()
        q_sum = sum_quadric10(self.quadrics10[a], self.quadrics10[b])
        cost = eval_quadric10(q_sum, float(position[0]), float(position[1]), float(position[2]))

        # Hook pour la saillance (Papier 3 - Limper 2016)
        if self.saliency_weight > 0.0 and self.vertex_saliency is not None:
            sal_a = self.vertex_saliency[a] if a < len(self.vertex_saliency) else 0.0
            sal_b = self.vertex_saliency[b] if b < len(self.vertex_saliency) else 0.0
            sal_edge = 0.5 * (sal_a + sal_b)
            cost *= (1.0 + self.saliency_weight * sal_edge)

        return cost

    def _visible_face_vertices(self, face):
        return (face.a, face.b, face.c)

    def collect_edges(self):
        """Construit les incidences sommet-arête-face du maillage courant."""
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
        """Retourne les faces incidentes à a ou b sans cloner le maillage."""
        face_indices = self.vertex_faces.get(a, set()) | self.vertex_faces.get(b, set())
        return [
            self.faces[index].clone()
            for index in sorted(face_indices)
            if self.faces[index].visible
        ]

    @staticmethod
    def _point_triangle_distance_squared(point, first, second, third):
        """Calcule la distance au carré entre un point et un triangle."""
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

    def _total_collapse_energy(self, a, b, position):
        """Retourne l'énergie totale de contraction."""
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

    def _candidate_position(self, a, b):
        """Calcule la position qui minimise l'erreur (QEM Garland 1997 ou Énergie Hoppe 1996)."""
        candidates = [
            self.vertices[a],
            self.vertices[b],
            (self.vertices[a] + self.vertices[b]) / 2.0
        ]

        best_position = candidates[0]
        best_cost = float("inf")
        for position in candidates:
            if self.use_qem:
                cost = self.qem_cost(a, b, position)
            else:
                cost = self._total_collapse_energy(a, b, position)

            if np.isfinite(cost) and cost < best_cost:
                best_position = position
                best_cost = cost

        return np.asarray(best_position, dtype=float), max(0.0, best_cost)

    def edge_length(self, a, b):
        """Retourne l'énergie totale de contraction de l'arête (a, b)."""
        if (a, b) not in self.edge_faces and (b, a) not in self.edge_faces:
            return float("inf")
        _, cost = self._candidate_position(a, b)
        return cost

    # Validité géométrique et topologique
    def _link_condition(self, vs, vt):
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
        """Vérifie si le sommet v appartient à au moins une arête de bord (1 seule face incidente)."""
        for nbr in self.vertex_neighbors.get(v, ()):
            edge = (v, nbr) if v < nbr else (nbr, v)
            faces = self.edge_faces.get(edge)
            if faces is not None and len(faces) == 1:
                return True
        return False

    def is_valid_collapse(self, vs, vt, candidate=None):
        """Applique les contrôles de bord, manifold, link et retournement."""
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

        affected = (
            self.vertex_faces.get(vs, set())
            | self.vertex_faces.get(vt, set())
        )
        for face_index in affected:
            face = self.faces[face_index]
            if not face.visible:
                continue

            old_vertices = self._visible_face_vertices(face)
            new_vertices = tuple(vt if vertex == vs else vertex for vertex in old_vertices)
            if len(set(new_vertices)) < 3:
                continue

            old_points = [self.vertices[index] for index in old_vertices]
            new_points = [
                candidate if index == vt else self.vertices[index]
                for index in new_vertices
            ]
            old_normal = np.cross(
                old_points[1] - old_points[0],
                old_points[2] - old_points[0]
            )
            new_normal = np.cross(
                new_points[1] - new_points[0],
                new_points[2] - new_points[0]
            )
            old_norm = np.linalg.norm(old_normal)
            new_norm = np.linalg.norm(new_normal)
            if new_norm <= 1e-12:
                return False
            if old_norm > 1e-12 and np.dot(old_normal, new_normal) <= 0:
                return False

        return True

    # File de priorité et contraction
    def _build_priority_queue(self):
        """Classe les arêtes par coût (QEM Garland 1997 ou longueur)."""
        self._heap_version += 1
        self._heap = []
        self._pending_candidate = None
        self._pending_edge = None
        if self.use_qem:
            if not self.quadrics10:
                self.compute_initial_quadrics()
            for edge in self.collect_edges():
                candidate, cost = self._candidate_position(*edge)
                heapq.heappush(self._heap, (cost, edge, self._heap_version))
        else:
            for edge in self.collect_edges():
                first, second = (self.vertices[index] for index in edge)
                priority = float(np.dot(first - second, first - second))
                heapq.heappush(self._heap, (priority, edge, self._heap_version))

    def shortest_edge(self):
        """Retourne l'arête valide d'énergie/erreur minimale."""
        if not self._heap:
            self._build_priority_queue()
        while self._heap:
            _, edge, version = heapq.heappop(self._heap)
            if version != self._heap_version:
                continue
            if edge not in self.edge_faces:
                continue
            candidate, cost = self._candidate_position(*edge)
            if not np.isfinite(cost):
                continue
            if self.is_valid_collapse(*edge, candidate=candidate):
                self._pending_edge = edge
                self._pending_candidate = (candidate, cost)
                return edge
        # Si le tas est épuisé, reconstruire une fois
        self._build_priority_queue()
        while self._heap:
            _, edge, version = heapq.heappop(self._heap)
            if version != self._heap_version or edge not in self.edge_faces:
                continue
            candidate, cost = self._candidate_position(*edge)
            if not np.isfinite(cost):
                continue
            if self.is_valid_collapse(*edge, candidate=candidate):
                self._pending_edge = edge
                self._pending_candidate = (candidate, cost)
                return edge
        return None

    def collapse_edge(self, vs, vt):
        """Contracte vs vers vt et enregistre toutes les données du split (mise à jour topologique locale O(1))."""
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
            add_quadric10_inplace(self.quadrics10[vt], self.quadrics10[vs])
            self.quadrics10[vs] = [0.0] * 10

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

        if self.use_qem and self._heap:
            for neighbor in self.vertex_neighbors.get(vt, ()):
                nbr_edge = (vt, neighbor) if vt < neighbor else (neighbor, vt)
                if nbr_edge in self.edge_faces:
                    nbr_cand, nbr_cost = self._candidate_position(*nbr_edge)
                    heapq.heappush(self._heap, (nbr_cost, nbr_edge, self._heap_version))

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

    # Export du maillage courant et de la séquence progressive
    def write_obja(self, output):
        """Écrit le niveau courant puis les opérations de reconstruction."""
        output_model = obja.Output(output, random_color=False)

        def ensure_vertex(vertex_id, position=None):
            if vertex_id not in output_model.vertex_mapping:
                if position is None:
                    position = self.vertices[vertex_id]
                output_model.add_vertex(vertex_id, position)

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

        emitted_faces = set()
        for face_index, face in enumerate(self.faces):
            if face.visible:
                output_model.add_face(face_index, face)
                emitted_faces.add(face_index)

        for collapse in reversed(self.history):
            ensure_vertex(collapse.vertex_id, collapse.vertex_position)
            ensure_vertex(
                collapse.target_vertex_id,
                collapse.target_vertex_position
            )
            output_model.edit_vertex(
                collapse.target_vertex_id,
                collapse.target_vertex_position
            )
            for face_index, old_face, _ in collapse.modified_faces:
                if face_index in emitted_faces:
                    ensure_face_vertices(old_face)
                    output_model.edit_face(face_index, old_face)
            for face_index, old_face in collapse.removed_faces:
                ensure_face_vertices(old_face)
                output_model.add_face(face_index, old_face)
                emitted_faces.add(face_index)

    @staticmethod
    def _validate_bits(bits):
        if not isinstance(bits, int) or not 1 <= bits <= 32:
            raise ValueError("bits doit être un entier compris entre 1 et 32.")

    @staticmethod
    def _quantize_positions(positions, bits):
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
        """Reconstruit des positions flottantes à partir de positions quantifiées."""
        ProgressiveMesh._validate_bits(bits)
        minimum, maximum = (
            np.asarray(value, dtype=float) for value in bounds
        )
        extent = np.where(maximum - minimum == 0, 1.0, maximum - minimum)
        scale = float((1 << bits) - 1)
        return minimum + np.asarray(data, dtype=float) / scale * extent

    def quantize_vertices(self, bits=16):
        """Quantifie puis déquantifie les vertices du maillage courant."""
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
        """Retourne les positions quantifiées de l'historique."""
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

    @staticmethod
    def _face_to_data(face):
        return [face.a, face.b, face.c, bool(face.visible)]

    @staticmethod
    def _face_from_data(data):
        return obja.Face(data[0], data[1], data[2], data[3])

    @staticmethod
    def _json_bytes(data):
        return json.dumps(
            data,
            separators=(",", ":"),
            sort_keys=True
        ).encode("utf-8")

    def encode_base_mesh(self, bits=16):
        """Sérialise le maillage courant et son historique."""
        self._validate_bits(bits)
        positions = list(self.vertices)
        for collapse in self.history:
            positions.extend([
                collapse.vertex_position,
                collapse.target_vertex_position
            ])
        _, minimum, maximum = self._quantize_positions(positions, bits)
        extent = np.where(
            np.asarray(maximum) - np.asarray(minimum) == 0,
            1.0,
            np.asarray(maximum) - np.asarray(minimum)
        )
        scale = (1 << bits) - 1

        def encode_position(position):
            return np.rint(
                (np.asarray(position) - minimum) / extent * scale
            ).astype(int).tolist()

        return {
            "version": 1,
            "bits": bits,
            "minimum": minimum,
            "maximum": maximum,
            "vertices": [encode_position(value) for value in self.vertices],
            "faces": [self._face_to_data(face) for face in self.faces],
            "history": [
                {
                    "vertex_id": collapse.vertex_id,
                    "vertex_position": encode_position(collapse.vertex_position),
                    "target_vertex_id": collapse.target_vertex_id,
                    "target_vertex_position": encode_position(
                        collapse.target_vertex_position
                    ),
                    "removed_faces": [
                        [index, self._face_to_data(face)]
                        for index, face in collapse.removed_faces
                    ],
                    "modified_faces": [
                        [
                            index,
                            self._face_to_data(old_face),
                            self._face_to_data(new_face)
                        ]
                        for index, old_face, new_face in collapse.modified_faces
                    ],
                    "cost": collapse.cost
                }
                for collapse in self.history
            ]
        }

    @staticmethod
    def compress_progressive_mesh(data):
        """Compresse une structure sérialisée avec zlib et un checksum SHA-256."""
        raw = ProgressiveMesh._json_bytes(data) if isinstance(data, dict) else bytes(data)
        checksum = hashlib.sha256(raw).digest()
        compressed = zlib.compress(raw, level=9)
        return b"PMZ1" + struct.pack(">Q", len(raw)) + checksum + compressed

    @staticmethod
    def _decompress_progressive_mesh(data):
        if (
            not isinstance(data, (bytes, bytearray))
            or len(data) < 44
            or data[:4] != b"PMZ1"
        ):
            raise ValueError("Format progressive mesh invalide.")
        raw_size = struct.unpack(">Q", data[4:12])[0]
        checksum = data[12:44]
        try:
            raw = zlib.decompress(data[44:])
        except zlib.error as error:
            raise ValueError("Données compressées invalides.") from error
        if len(raw) != raw_size or hashlib.sha256(raw).digest() != checksum:
            raise ValueError("Checksum ou taille du fichier invalide.")
        return json.loads(raw.decode("utf-8"))

    def write_progressive_binary(self, path, bits=16):
        """Écrit un progressive mesh quantifié et compressé."""
        payload = self.encode_base_mesh(bits)
        compressed = self.compress_progressive_mesh(payload)
        with open(path, "wb") as file:
            file.write(compressed)
        return len(compressed)

    @classmethod
    def read_progressive_binary(cls, path):
        """Charge un progressive mesh depuis un fichier compressé."""
        with open(path, "rb") as file:
            payload = cls._decompress_progressive_mesh(file.read())
        bits = payload["bits"]
        bounds = (payload["minimum"], payload["maximum"])
        model = cls()
        model.vertices = [
            value for value in cls.dequantize_vertices(
                payload["vertices"],
                bounds,
                bits
            )
        ]
        model.faces = [
            cls._face_from_data(face) for face in payload["faces"]
        ]
        for item in payload["history"]:
            positions = cls.dequantize_vertices(
                [item["vertex_position"], item["target_vertex_position"]],
                bounds,
                bits
            )
            removed_faces = [
                (index, cls._face_from_data(face))
                for index, face in item["removed_faces"]
            ]
            modified_faces = [
                (
                    index,
                    cls._face_from_data(old_face),
                    cls._face_from_data(new_face)
                )
                for index, old_face, new_face in item["modified_faces"]
            ]
            model.history.append(
                Collapse(
                    item["vertex_id"],
                    positions[0],
                    item["target_vertex_id"],
                    positions[1],
                    removed_faces,
                    modified_faces,
                    item["cost"]
                )
            )
        model.collect_edges()
        return model

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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Progressive Mesh avec Quadric Error Metrics (Garland & Heckbert 1997)."
    )
    parser.add_argument("model", nargs="?", default="example/bunny.obj", help="Chemin vers le modèle OBJ (défaut: example/bunny.obj)")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio de faces cibles (ex: 0.1 pour 10%%)")
    parser.add_argument("--method", choices=["qem", "hoppe"], default="qem", help="Méthode de coût d'erreur (qem ou hoppe)")
    parser.add_argument("--boundary-weight", type=float, default=10.0, help="Poids des plans de contrainte de bord (Garland §6)")
    parser.add_argument("--saliency-weight", type=float, default=0.0, help="Poids de la saillance (hook prêt pour papier 3)")
    parser.add_argument("--bits", type=int, default=16, help="Bits de quantification pour le binaire PMZ (défaut: 16)")
    parser.add_argument("--output-obja", type=str, default=None, help="Chemin de sortie OBJA personnalisé")
    parser.add_argument("--output-pmz", type=str, default=None, help="Chemin de sortie PMZ personnalisé")
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

    print("=" * 70)
    print(f"  PROGRESSIVE MESH — {'QEM (Garland & Heckbert 1997)' if args.method == 'qem' else 'Énergie Hoppe 1996'}")
    print("=" * 70)
    print(f"Modèle d'entrée    : {model_path.name}")
    print(f"Méthode choisie    : {args.method.upper()}")
    print(f"Ratio cible        : {args.ratio:.1%}")

    # 1. Chargement du modèle
    model = ProgressiveMesh(
        use_qem=(args.method == "qem"),
        boundary_weight=args.boundary_weight,
        saliency_weight=args.saliency_weight,
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
    print(f"\n[2] Simplification progressive en cours ({args.method.upper()})...")
    t0_sim = perf_counter()
    model.simplify(target_ratio=args.ratio)
    t_sim = perf_counter() - t0_sim

    final_faces = model.count_visible_faces()
    num_collapses = len(model.history)
    rate = num_collapses / max(1e-6, t_sim)
    print(f"    - Terminé en {t_sim:.4f}s ({rate:.0f} collapses/sec) !")
    print(f"    - Faces finales : {initial_faces} -> {final_faces} ({final_faces / initial_faces:.1%})")
    print(f"    - Vertex splits enregistrés : {num_collapses}")

    # 3. Export OBJA
    stem = f"{model_path.stem}_{args.method}_r{args.ratio:g}"
    obja_path = Path(args.output_obja) if args.output_obja else Path("example") / f"{stem}.obja"
    obja_path.parent.mkdir(parents=True, exist_ok=True)
    with open(obja_path, "w", encoding="utf-8") as file:
        model.write_obja(file)
    obja_bytes = obja_path.stat().st_size
    print(f"\n[3] Export OBJA progressif :")
    print(f"    - Fichier OBJA : {obja_path.name} ({obja_bytes:,} octets, {obja_bytes / 1024:.1f} Ko)")

    # 4. Quantification et compression binaire PMZ
    pmz_path = Path(args.output_pmz) if args.output_pmz else Path("example") / f"{stem}.pmz"
    pmz_bytes = model.write_progressive_binary(str(pmz_path), bits=args.bits)
    orig_bytes = model_path.stat().st_size
    ratio_comp = pmz_bytes / orig_bytes if orig_bytes > 0 else 0.0
    gain = orig_bytes / max(1, pmz_bytes)
    print(f"\n[4] Quantification ({args.bits} bits) et Compression Binaire PMZ :")
    print(f"    - Fichier PMZ compressé (zlib) : {pmz_path.name} ({pmz_bytes:,} octets, {pmz_bytes / 1024:.1f} Ko)")
    print(f"    - Taux de compression vs OBJ original : {ratio_comp:.1%} (taille divisée par {gain:.1f}x)")

    # 5. Validation et test de réversibilité
    restored = ProgressiveMesh.read_progressive_binary(str(pmz_path))
    num_restored = len(restored.history)
    exact_splits = num_restored == num_collapses
    split_ok = restored.vertex_split()
    print(f"\n[5] Test de relecture et décodage du PMZ :")
    print(f"    - Collapses relus avec succès : {num_restored} / {num_collapses} ({'OK' if exact_splits else 'ERREUR'})")
    print(f"    - Test d'un vertex split réversible : {'SUCCÈS' if split_ok else 'ÉCHEC'}")

    # 6. Métriques de comparaison géométrique vs original
    try:
        comp = comparaison.compare_files(str(model_path), str(obja_path))
        print(f"\n[6] Métriques d'erreur géométrique vs original (Hausdorff & Moyenne) :")
        print(f"    - Sommets restants               : {comp['original_vertices']} -> {comp['final_vertices']} (-{comp['vertex_reduction_percent']:.1f}%)")
        print(f"    - Faces restantes                : {comp['original_faces']} -> {comp['final_faces']} (-{comp['face_reduction_percent']:.1f}%)")
        print(f"    - Erreur géométrique moyenne     : {comp['mean_geometric_error']:.6f}")
        print(f"    - Erreur géométrique maximale    : {comp['maximum_geometric_error']:.6f} (Hausdorff)")
        print(f"    - Erreur moyenne normalisée      : {comp['normalized_mean_error_percent']:.4f}% de la diagonale")
        print(f"    - Erreur maximale normalisée     : {comp['normalized_maximum_error_percent']:.4f}% de la diagonale")
    except Exception as e:
        print(f"\n[6] Erreur lors du calcul comparatif : {e}")

    print("=" * 70)


if __name__ == "__main__":
    main()
