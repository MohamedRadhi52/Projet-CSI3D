#!/usr/bin/env python3
"""
Module Multiresolution Streaming Mesh — Chen & Nishita (Web3D 2002).

Implémente la simplification par Half-Edge Collapse avec préservation de forme
(sharp edges, seuils multi-niveaux et choix du sommet conservé selon la
déviation des vecteurs normaux des faces adjacentes).
"""

import math
import numpy as np


def compute_face_normals(vertices, faces):
    """Calcule le dictionnaire {face_index: (nx, ny, nz)} unitaire pour les faces visibles."""
    normals = {}
    for face_index, face in enumerate(faces):
        if not getattr(face, "visible", True):
            continue
        p0 = vertices[face.a]
        p1 = vertices[face.b]
        p2 = vertices[face.c]
        ux = float(p1[0] - p0[0])
        uy = float(p1[1] - p0[1])
        uz = float(p1[2] - p0[2])
        vx = float(p2[0] - p0[0])
        vy = float(p2[1] - p0[1])
        vz = float(p2[2] - p0[2])
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm > 1e-12:
            inv = 1.0 / norm
            normals[face_index] = (nx * inv, ny * inv, nz * inv)
    return normals


def vertex_normal_deviation(vertex_id, vertex_faces, face_normals):
    """
    Mesure la déviation des normales des faces incidentes à vertex_id
    (Chen & Nishita 2002, §4.2 Fig. 6).
    Plus les faces autour du sommet forment un angle vif, plus le score est élevé.
    """
    sx = sy = sz = 0.0
    count = 0
    for fid in vertex_faces.get(vertex_id, ()):
        n = face_normals.get(fid)
        if n is not None:
            sx += n[0]
            sy += n[1]
            sz += n[2]
            count += 1
    if count <= 1:
        return 0.0

    mn = math.sqrt(sx * sx + sy * sy + sz * sz)
    if mn <= 1e-12:
        return float(count)
    inv = 1.0 / mn
    mx, my, mz = sx * inv, sy * inv, sz * inv

    dev = 0.0
    for fid in vertex_faces.get(vertex_id, ()):
        n = face_normals.get(fid)
        if n is not None:
            dot = n[0] * mx + n[1] * my + n[2] * mz
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0
            dev += 1.0 - dot
    return dev


def edge_normal_weight(edge, edge_faces, vertices, face_normals):
    """
    Calcule le poids d'une arête d'après la différence entre les normales
    de ses deux faces adjacentes (Chen & Nishita 2002, §3-4).
    """
    incident = edge_faces.get(edge)
    if not incident or len(incident) != 2:
        return float("inf")

    it = iter(incident)
    n0 = face_normals.get(next(it))
    n1 = face_normals.get(next(it))
    if n0 is None or n1 is None:
        return float("inf")

    dx = n0[0] - n1[0]
    dy = n0[1] - n1[1]
    dz = n0[2] - n1[2]
    normal_diff = math.sqrt(dx * dx + dy * dy + dz * dz)

    p0, p1 = vertices[edge[0]], vertices[edge[1]]
    ex = float(p1[0] - p0[0])
    ey = float(p1[1] - p0[1])
    ez = float(p1[2] - p0[2])
    edge_len = math.sqrt(ex * ex + ey * ey + ez * ez)
    return normal_diff + 1e-4 * edge_len


def classify_unremovable_vertices(
    edge_faces,
    vertices,
    face_normals,
    level_threshold,
    sharp_threshold=0.85,
    lock_sharp_edges=True,
):
    """
    Identifie les sommets non supprimables au niveau courant (Chen & Nishita §3-4) :
    - extrémités des arêtes de bord (len == 1)
    - extrémités des sharp edges (écart de normale >= sharp_threshold) si lock_sharp_edges
    - extrémités des base edges du niveau courant (poids > level_threshold)
    """
    unremovable = set()
    for edge, incident in edge_faces.items():
        if len(incident) != 2:
            unremovable.update(edge)
            continue

        it = iter(incident)
        n0 = face_normals.get(next(it))
        n1 = face_normals.get(next(it))
        if n0 is None or n1 is None:
            continue

        dx = n0[0] - n1[0]
        dy = n0[1] - n1[1]
        dz = n0[2] - n1[2]
        normal_diff = math.sqrt(dx * dx + dy * dy + dz * dz)
        if (lock_sharp_edges and normal_diff >= sharp_threshold) or (normal_diff > level_threshold):
            unremovable.update(edge)

    return unremovable


def choose_half_edge_direction(
    a,
    b,
    edge_faces,
    vertices,
    vertex_faces,
    face_normals,
    unremovable_vertices=None,
):
    """
    Applique la politique Half-Edge Collapse de Chen & Nishita (§4.2) sur (a, b) :
    - Retourne (vs_remove, vt_keep, kept_position, cost) si au moins un sommet est supprimable.
    - Si vt est un-removable ou a une plus grande déviation de normales, on supprime vs -> vt.
    - La position du sommet conservé vt reste inchangée (aucun 'ev' nécessaire).
    """
    edge = (a, b) if a < b else (b, a)
    weight = edge_normal_weight(edge, edge_faces, vertices, face_normals)
    if not math.isfinite(weight):
        return None

    if unremovable_vertices:
        a_locked = a in unremovable_vertices
        b_locked = b in unremovable_vertices
    else:
        a_locked = b_locked = False

    if a_locked and b_locked:
        return None
    if b_locked and not a_locked:
        vs, vt = a, b
    elif a_locked and not b_locked:
        vs, vt = b, a
    else:
        dev_a = vertex_normal_deviation(a, vertex_faces, face_normals)
        dev_b = vertex_normal_deviation(b, vertex_faces, face_normals)
        if dev_b >= dev_a:
            vs, vt = a, b
        else:
            vs, vt = b, a

    return vs, vt, np.copy(vertices[vt]), weight
