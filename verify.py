
import numpy as np
import obja


ORIGINAL_FILE = "example/suzanne.obj"
RECONSTRUCTED_FILE = "example/suzanne_pm.obja"

TOLERANCE = 1e-8


def face_tuple(face):
    return (face.a, face.b, face.c)


def canonical_face(face):
    return tuple(sorted(face_tuple(face)))


def check_indices(model):
    invalid = []
    degenerate = []

    for i, face in enumerate(model.faces):
        indices = face_tuple(face)

        if min(indices) < 0 or max(indices) >= len(model.vertices):
            invalid.append(i)
            continue

        if len(set(indices)) != 3:
            degenerate.append(i)

    return invalid, degenerate


def compare_vertices(original, reconstructed):
    if len(original.vertices) != len(reconstructed.vertices):
        return False, None, None

    original_vertices = np.asarray(original.vertices, dtype=float)
    reconstructed_vertices = np.asarray(
        reconstructed.vertices,
        dtype=float
    )

    if original_vertices.shape != reconstructed_vertices.shape:
        return False, None, None

    differences = np.linalg.norm(
        original_vertices - reconstructed_vertices,
        axis=1
    )

    max_difference = float(np.max(differences))
    mean_difference = float(np.mean(differences))

    return (
        np.allclose(
            original_vertices,
            reconstructed_vertices,
            atol=TOLERANCE,
            rtol=0
        ),
        max_difference,
        mean_difference
    )


def compare_faces(original, reconstructed):
    if len(original.faces) != len(reconstructed.faces):
        return False, False, 0, 0

    original_ordered = [
        face_tuple(face)
        for face in original.faces
    ]

    reconstructed_ordered = [
        face_tuple(face)
        for face in reconstructed.faces
    ]

    ordered_ok = original_ordered == reconstructed_ordered

    original_canonical = [
        canonical_face(face)
        for face in original.faces
    ]

    reconstructed_canonical = [
        canonical_face(face)
        for face in reconstructed.faces
    ]

    unordered_ok = sorted(original_canonical) == sorted(
        reconstructed_canonical
    )

    original_set = set(original_canonical)
    reconstructed_set = set(reconstructed_canonical)

    missing = original_set - reconstructed_set
    extra = reconstructed_set - original_set

    return (
        ordered_ok,
        unordered_ok,
        len(missing),
        len(extra)
    )


def main():
    print("Chargement des fichiers...")

    original = obja.parse_file(ORIGINAL_FILE)
    reconstructed = obja.parse_file(RECONSTRUCTED_FILE)

    print("\n========== COMPTAGE ==========")

    print("Sommets originaux :", len(original.vertices))
    print("Sommets reconstruits :", len(reconstructed.vertices))

    print("Faces originales :", len(original.faces))
    print("Faces reconstruites :", len(reconstructed.faces))

    vertex_count_ok = (
        len(original.vertices) == len(reconstructed.vertices)
    )

    face_count_ok = (
        len(original.faces) == len(reconstructed.faces)
    )

    print("\n========== COORDONNÉES ==========")

    vertices_ok, max_diff, mean_diff = compare_vertices(
        original,
        reconstructed
    )

    print("Nombre de sommets identique :", vertex_count_ok)

    if max_diff is not None:
        print("Différence maximale :", f"{max_diff:.12e}")
        print("Différence moyenne :", f"{mean_diff:.12e}")

    print("Coordonnées identiques :", vertices_ok)

    print("\n========== TOPOLOGIE ==========")

    ordered_ok, unordered_ok, missing, extra = compare_faces(
        original,
        reconstructed
    )

    print("Nombre de faces identique :", face_count_ok)
    print("Faces identiques dans le même ordre :", ordered_ok)
    print("Topologie identique indépendamment de l'ordre :", unordered_ok)
    print("Faces manquantes :", missing)
    print("Faces supplémentaires :", extra)

    print("\n========== VALIDITÉ ==========")

    original_invalid, original_degenerate = check_indices(original)
    reconstructed_invalid, reconstructed_degenerate = (
        check_indices(reconstructed)
    )

    print("Indices invalides originaux :", len(original_invalid))
    print("Indices invalides reconstruits :", len(reconstructed_invalid))

    print("Faces dégénérées originales :", len(original_degenerate))
    print("Faces dégénérées reconstruites :", len(reconstructed_degenerate))

    valid_ok = (
        len(reconstructed_invalid) == 0
        and len(reconstructed_degenerate) == 0
    )

    print("\n========== RÉSULTAT FINAL ==========")

    exact_ordered = (
        vertex_count_ok
        and face_count_ok
        and vertices_ok
        and ordered_ok
        and valid_ok
    )

    exact_topology = (
        vertex_count_ok
        and face_count_ok
        and vertices_ok
        and unordered_ok
        and valid_ok
    )

    print("Reconstruction exacte avec ordre :", exact_ordered)
    print("Reconstruction exacte sans ordre :", exact_topology)

    if exact_ordered:
        print("\nSUCCÈS : reconstruction exactement identique.")
    elif exact_topology:
        print("\nSUCCÈS : même géométrie et même topologie, ordre différent.")
    else:
        print("\nÉCHEC : la reconstruction diffère du modèle original.")


if __name__ == "__main__":
    main()
