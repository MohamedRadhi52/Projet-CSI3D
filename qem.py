#!/usr/bin/env python3
"""
Module Quadric Error Metrics (QEM) — Garland & Heckbert (SIGGRAPH 1997).
Contient toutes les structures et fonctions nécessaires pour le calcul
des quadriques d'erreur 4x4 symétriques (10 flottants), des contraintes
de bord (Garland §6), et de la sélection de la position optimale d'arête.
"""

import numpy as np


# ----------------------------------------------------------------------
# Représentation compacte symétrique 4x4 d'une quadrique en 10 flottants :
# Q = [ q0  q1  q2  q3 ]
#     [ q1  q4  q5  q6 ]
#     [ q2  q5  q7  q8 ]
#     [ q3  q6  q8  q9 ]
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
    """Évalue v^T Q v via la formule développée de Garland & Heckbert (1997, §4 note 2)."""
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


# ----------------------------------------------------------------------
# Initialisation des quadriques de sommets du maillage
# ----------------------------------------------------------------------

def compute_initial_quadrics(vertices, faces, edge_faces, boundary_weight: float = 10.0) -> list[list[float]]:
    """
    Initialise les quadriques 10-float par sommet selon Garland & Heckbert (1997) :
    1. Somme des quadriques des faces incidentes pondérées par leur aire.
    2. Plans de contrainte perpendiculaires sur les arêtes de frontière (Garland §6).
    """
    num_v = len(vertices)
    quadrics10 = [[0.0] * 10 for _ in range(num_v)]

    # 1. Quadriques des faces (pondérées par l'aire du triangle)
    for face in faces:
        if not face.visible:
            continue
        p0 = vertices[face.a]
        p1 = vertices[face.b]
        p2 = vertices[face.c]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        normal /= norm
        area = 0.5 * norm
        d = -float(np.dot(normal, p0))
        kq = plane_to_quadric10(normal[0], normal[1], normal[2], d, weight=area)
        add_quadric10_inplace(quadrics10[face.a], kq)
        add_quadric10_inplace(quadrics10[face.b], kq)
        add_quadric10_inplace(quadrics10[face.c], kq)

    # 2. Plans de contrainte perpendiculaires sur les bords (Garland §6)
    if boundary_weight > 0.0:
        for edge, incident_faces in edge_faces.items():
            if len(incident_faces) == 1:
                u, w = edge
                pu, pw = vertices[u], vertices[w]
                edge_vec = pw - pu
                edge_len = float(np.linalg.norm(edge_vec))
                if edge_len <= 1e-12:
                    continue
                face_idx = next(iter(incident_faces))
                f = faces[face_idx]
                fn = np.cross(vertices[f.b] - vertices[f.a], vertices[f.c] - vertices[f.a])
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
                kq_bnd = plane_to_quadric10(perp[0], perp[1], perp[2], d_perp, weight=boundary_weight * edge_len)
                add_quadric10_inplace(quadrics10[u], kq_bnd)
                add_quadric10_inplace(quadrics10[w], kq_bnd)

    return quadrics10


# ----------------------------------------------------------------------
# Évaluation d'erreur et sélection de la position optimale
# ----------------------------------------------------------------------

def eval_edge_cost(quadrics10, a: int, b: int, position, saliency_weight: float = 0.0, vertex_saliency=None) -> float:
    """
    Évalue le coût quadrique QEM v^T (Qa + Qb) v d'une position candidate,
    avec pondération facultative par la saillance (Limper 2016).
    """
    q_sum = sum_quadric10(quadrics10[a], quadrics10[b])
    cost = eval_quadric10(q_sum, float(position[0]), float(position[1]), float(position[2]))

    # Pondération par la saillance si activée
    if saliency_weight > 0.0 and vertex_saliency is not None:
        sal_a = vertex_saliency[a] if a < len(vertex_saliency) else 0.0
        sal_b = vertex_saliency[b] if b < len(vertex_saliency) else 0.0
        sal_edge = 0.5 * (sal_a + sal_b)
        cost *= (1.0 + saliency_weight * sal_edge)

    return cost


def optimal_candidate_position(quadrics10, vertices, a: int, b: int, saliency_weight: float = 0.0, vertex_saliency=None):
    """
    Trouve la position candidate minimisant l'erreur quadrique QEM parmi :
    v_a, v_b, ou le milieu (v_a + v_b) / 2.
    Retourne (best_position, best_cost).
    """
    candidates = [
        vertices[a],
        vertices[b],
        (vertices[a] + vertices[b]) / 2.0
    ]

    best_position = candidates[0]
    best_cost = float("inf")
    for position in candidates:
        cost = eval_edge_cost(quadrics10, a, b, position, saliency_weight, vertex_saliency)
        if np.isfinite(cost) and cost < best_cost:
            best_position = position
            best_cost = cost

    return np.asarray(best_position, dtype=float), max(0.0, best_cost)


def update_quadrics_on_collapse(quadrics10, vs: int, vt: int) -> None:
    """
    Met à jour les quadriques lors du collapse de vs vers vt :
    Q_t = Q_t + Q_s, puis Q_s est remis à zéro.
    """
    add_quadric10_inplace(quadrics10[vt], quadrics10[vs])
    quadrics10[vs] = [0.0] * 10
