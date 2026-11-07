#!/usr/bin/env python3

import hashlib
import heapq
import json
import struct
import sys
import zlib

import numpy as np

import obja
import comparaison


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

    def __init__(self):
        super().__init__()
        self.history = []
        self.deleted_faces = set()
        self.edge_faces = {}
        self.vertex_neighbors = {}
        self.vertex_faces = {}
        self.vertex_quadrics = []
        self._heap = []
        self._heap_version = 0

    # Structures de voisinage et quadrics
    def _visible_face_vertices(self, face):
        return (face.a, face.b, face.c)

    def collect_edges(self):
        """Construit les incidences sommet-arête-face du maillage courant."""
        self.edge_faces = {}
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

    def _rebuild_quadrics(self):
        self.vertex_quadrics = [
            np.zeros((4, 4), dtype=float)
            for _ in self.vertices
        ]

        for face in self.faces:
            if not face.visible:
                continue

            a, b, c = self._visible_face_vertices(face)
            p1, p2, p3 = self.vertices[a], self.vertices[b], self.vertices[c]
            normal = np.cross(p2 - p1, p3 - p1)
            norm = np.linalg.norm(normal)
            if norm <= 1e-12:
                continue

            normal /= norm
            plane = np.array(
                [normal[0], normal[1], normal[2], -np.dot(normal, p1)],
                dtype=float
            )
            quadric = np.outer(plane, plane)
            self.vertex_quadrics[a] += quadric
            self.vertex_quadrics[b] += quadric
            self.vertex_quadrics[c] += quadric

    def _candidate_position(self, a, b):
        """Calcule le minimum QEM, avec repli sûr sur les positions connues."""
        quadric = self.vertex_quadrics[a] + self.vertex_quadrics[b]
        system = quadric[:3, :3]
        rhs = -quadric[:3, 3]

        candidates = [
            self.vertices[a],
            self.vertices[b],
            (self.vertices[a] + self.vertices[b]) / 2.0
        ]
        try:
            if abs(np.linalg.det(system)) > 1e-12:
                candidates.insert(0, np.linalg.solve(system, rhs))
        except np.linalg.LinAlgError:
            pass

        best_position = candidates[0]
        best_cost = float("inf")
        for position in candidates:
            homogeneous = np.append(position, 1.0)
            cost = float(homogeneous @ quadric @ homogeneous)
            if np.isfinite(cost) and cost < best_cost:
                best_position = position
                best_cost = cost

        return np.asarray(best_position, dtype=float), max(0.0, best_cost)

    def edge_length(self, a, b):
        """Retourne le coût quadratique de contraction QEM de (a, b)."""
        if not self.vertex_quadrics:
            self._rebuild_quadrics()
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

    def is_valid_collapse(self, vs, vt, candidate=None):
        """Applique les contrôles de bord, manifold, link et retournement."""
        if vs == vt or not (0 <= vs < len(self.vertices)) or not (0 <= vt < len(self.vertices)):
            return False
        if tuple(sorted((vs, vt))) not in self.edge_faces:
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
        self._heap_version += 1
        self._heap = []
        for edge in self.collect_edges():
            cost = self.edge_length(*edge)
            if np.isfinite(cost):
                heapq.heappush(self._heap, (cost, edge, self._heap_version))

    def shortest_edge(self):
        """Retourne l'arête valide de coût QEM minimal."""
        self._rebuild_quadrics()
        self._build_priority_queue()
        while self._heap:
            cost, edge, version = heapq.heappop(self._heap)
            if version != self._heap_version:
                continue
            candidate, _ = self._candidate_position(*edge)
            if self.is_valid_collapse(*edge, candidate=candidate):
                return edge
        return None

    def collapse_edge(self, vs, vt):
        """Contracte vs vers vt et enregistre toutes les données du split."""
        self.collect_edges()
        self._rebuild_quadrics()
        candidate, cost = self._candidate_position(vs, vt)
        if not self.is_valid_collapse(vs, vt, candidate=candidate):
            return False

        removed_faces = []
        modified_faces = []
        target_position = np.copy(self.vertices[vt])
        affected = sorted(
            self.vertex_faces.get(vs, set())
            | self.vertex_faces.get(vt, set())
        )

        for face_index in affected:
            face = self.faces[face_index]
            if not face.visible:
                continue

            old_face = face.clone()
            new_vertices = [
                vt if vertex == vs else vertex
                for vertex in self._visible_face_vertices(face)
            ]
            if len(set(new_vertices)) < 3:
                removed_faces.append((face_index, old_face))
                face.visible = False
                self.deleted_faces.add(face_index)
                continue

            face.a, face.b, face.c = new_vertices
            modified_faces.append((face_index, old_face, face.clone()))

        self.vertices[vt] = candidate
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
        self.collect_edges()
        return True

    # Progressive mesh et vertex split
    def simplify(self, target_ratio=0.1):
        if not 0 < target_ratio <= 1:
            raise ValueError("target_ratio doit être compris entre 0 et 1.")

        initial_faces = self.count_visible_faces()
        target_faces = max(1, int(initial_faces * target_ratio))
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

        self.history.pop()
        self.collect_edges()
        return True

    def count_visible_faces(self):
        return sum(face.visible for face in self.faces)

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
        output_model = obja.Output(output, random_color=True)

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
        raw = zlib.decompress(data[44:])
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


def main(target_ratio=0.98):
    model = ProgressiveMesh()
    model.parse_file("example/suzanne.obj")
    model.simplify(target_ratio=target_ratio)
    with open("example/suzanne_pm.obja", "w") as file:
        model.write_obja(file)
    comparison = comparaison.compare_models(
        obja.parse_file("example/suzanne.obj"),
        model
    )
    comparaison.print_comparison(comparison)
    compressed_size = model.write_progressive_binary(
        "example/suzanne.pmz",
        bits=16
    )

    restored = ProgressiveMesh.read_progressive_binary("example/suzanne.pmz")
    restored_faces = restored.count_visible_faces()
    restored_collapses = len(restored.history)
    with open("example/suzanne_decompressed.obja", "w") as file:
        restored.write_obja(file)
    split_result = restored.vertex_split()

    print("Collapses:", len(model.history))
    print("Fichier PMZ:", compressed_size, "octets")
    print("Faces relues:", restored_faces)
    print("Collapses relus:", restored_collapses)
    print("Vertex split:", split_result)
    print("Faces apres split:", restored.count_visible_faces())


if __name__ == "__main__":
    ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.98
    main(target_ratio=ratio)
