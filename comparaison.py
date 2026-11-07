#!/usr/bin/env python3

import argparse
import os

import numpy as np

import obja


def _visible_faces(model):
    return [face for face in model.faces if face.visible]


def _load_mesh(path, base_level_only=False):
    """Charge les instructions de sommet et de face d'un fichier OBJ/OBJA."""
    model = obja.Model()
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            if not values or values[0].startswith("#"):
                continue
            instruction = values[0]
            if instruction == "v" and len(values) >= 4:
                model.vertices.append(
                    np.array([float(values[1]), float(values[2]), float(values[3])])
                )
            elif instruction == "f" and len(values) >= 4:
                model.faces.append(
                    obja.Face(
                        _face_index(values[1]),
                        _face_index(values[2]),
                        _face_index(values[3])
                    )
                )
            elif instruction == "ev" and len(values) >= 5:
                if base_level_only:
                    break
                vertex_index = int(values[1]) - 1
                model.vertices[vertex_index] = np.array(
                    [float(values[2]), float(values[3]), float(values[4])]
                )
            elif instruction == "ef" and len(values) >= 5:
                if base_level_only:
                    break
                face_index = int(values[1]) - 1
                model.faces[face_index] = obja.Face(
                    _face_index(values[2]),
                    _face_index(values[3]),
                    _face_index(values[4])
                )
    return model


def _face_index(value):
    return int(value.split("/")[0]) - 1


def _vertices_as_array(model):
    if not model.vertices:
        return np.empty((0, 3), dtype=float)
    return np.asarray(model.vertices, dtype=float)


def _nearest_distances(source, target):
    if len(source) == 0 or len(target) == 0:
        return np.empty(0, dtype=float)
    distances = np.linalg.norm(
        source[:, None, :] - target[None, :, :],
        axis=2
    )
    return np.min(distances, axis=1)


def compare_models(original, final):
    """Compare deux modèles OBJA et retourne des métriques numériques."""
    original_vertices = _vertices_as_array(original)
    final_vertices = _vertices_as_array(final)
    original_faces = _visible_faces(original)
    final_faces = _visible_faces(final)

    original_to_final = _nearest_distances(original_vertices, final_vertices)
    final_to_original = _nearest_distances(final_vertices, original_vertices)
    all_vertices = np.concatenate(
        (original_vertices, final_vertices),
        axis=0
    )
    if len(all_vertices):
        diagonal = float(
            np.linalg.norm(
                np.max(all_vertices, axis=0) - np.min(all_vertices, axis=0)
            )
        )
    else:
        diagonal = 0.0

    maximum_error = max(
        float(np.max(original_to_final)) if len(original_to_final) else 0.0,
        float(np.max(final_to_original)) if len(final_to_original) else 0.0
    )
    mean_error = float(
        np.mean(np.concatenate((original_to_final, final_to_original)))
    ) if len(original_to_final) or len(final_to_original) else 0.0

    return {
        "original_vertices": len(original_vertices),
        "final_vertices": len(final_vertices),
        "original_faces": len(original_faces),
        "final_faces": len(final_faces),
        "vertex_reduction_percent": _reduction_percent(
            len(original_vertices),
            len(final_vertices)
        ),
        "face_reduction_percent": _reduction_percent(
            len(original_faces),
            len(final_faces)
        ),
        "mean_geometric_error": mean_error,
        "maximum_geometric_error": maximum_error,
        "normalized_mean_error_percent": (
            100.0 * mean_error / diagonal if diagonal else 0.0
        ),
        "normalized_maximum_error_percent": (
            100.0 * maximum_error / diagonal if diagonal else 0.0
        ),
    }


def _reduction_percent(original_count, final_count):
    if original_count == 0:
        return 0.0
    return 100.0 * (original_count - final_count) / original_count


def compare_files(original_path, final_path):
    """Charge deux fichiers OBJ/OBJA et compare leurs maillages."""
    original = _load_mesh(original_path)
    final = _load_mesh(final_path, base_level_only=True)
    result = compare_models(original, final)
    result["original_file_size"] = os.path.getsize(original_path)
    result["final_file_size"] = os.path.getsize(final_path)
    result["file_size_reduction_percent"] = _reduction_percent(
        result["original_file_size"],
        result["final_file_size"]
    )
    return result


def print_comparison(result):
    """Affiche les résultats d'une comparaison dans le terminal."""
    print("Comparaison original / final")
    print("Sommets:", result["original_vertices"], "->", result["final_vertices"])
    print("Faces:", result["original_faces"], "->", result["final_faces"])
    print(
        "Reduction sommets: {:.2f}%".format(
            result["vertex_reduction_percent"]
        )
    )
    print(
        "Reduction faces: {:.2f}%".format(
            result["face_reduction_percent"]
        )
    )
    print(
        "Erreur geometrique moyenne: {:.6f}".format(
            result["mean_geometric_error"]
        )
    )
    print(
        "Erreur geometrique maximale: {:.6f}".format(
            result["maximum_geometric_error"]
        )
    )
    print(
        "Erreur moyenne normalisee: {:.4f}%".format(
            result["normalized_mean_error_percent"]
        )
    )
    print(
        "Erreur maximale normalisee: {:.4f}%".format(
            result["normalized_maximum_error_percent"]
        )
    )
    if "original_file_size" in result:
        print(
            "Taille fichiers: {} -> {} octets".format(
                result["original_file_size"],
                result["final_file_size"]
            )
        )
        print(
            "Reduction taille fichiers: {:.2f}%".format(
                result["file_size_reduction_percent"]
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare deux fichiers de maillage OBJ ou OBJA."
    )
    parser.add_argument("original", help="Fichier du modele original")
    parser.add_argument("final", help="Fichier du modele final")
    args = parser.parse_args()
    print_comparison(compare_files(args.original, args.final))


if __name__ == "__main__":
    main()
