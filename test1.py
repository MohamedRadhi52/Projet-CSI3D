
#!/usr/bin/env python3

import heapq
import sys
import numpy as np
import obja


class ProgressiveMesh(obja.Model):

    def __init__(self):
        super().__init__()

        self.active_faces = []
        self.quadrics = []
        self.operations = []

        self.neighbors = []
        self.incident_faces = []
        self.edge_faces = {}

        self.edge_versions = {}
        self.heap = []
        self.heap_counter = 0

        self.boundary_weight = 10.0
        self.feature_weight = 5.0
        self.feature_angle = 45.0

    def initialize(self):
        self.vertices = [
            np.asarray(v, dtype=float).copy()
            for v in self.vertices
        ]

        self.active_faces = [bool(f.visible) for f in self.faces]
        self.operations = []
        self.edge_versions = {}
        self.heap = []
        self.heap_counter = 0

        self.build_topology()
        self.compute_quadrics()

    def compute_quadrics(self):
        self.quadrics = [
            np.zeros((4, 4), dtype=float)
            for _ in self.vertices
        ]

        face_normals = {}
        edges = {}

        for i, face in enumerate(self.faces):
            if not self.active_faces[i]:
                continue

            a, b, c = face.a, face.b, face.c
            p1, p2, p3 = self.vertices[a], self.vertices[b], self.vertices[c]

            normal = np.cross(p2 - p1, p3 - p1)
            length = np.linalg.norm(normal)

            if length < 1e-12:
                continue

            normal /= length
            face_normals[i] = normal

            plane = np.append(normal, -np.dot(normal, p1))
            quadric = np.outer(plane, plane)

            self.quadrics[a] += quadric
            self.quadrics[b] += quadric
            self.quadrics[c] += quadric

            for u, v in ((a, b), (b, c), (c, a)):
                edge = (min(u, v), max(u, v))
                edges.setdefault(edge, []).append(i)

        threshold = np.cos(np.radians(self.feature_angle))

        for (u, v), incident in edges.items():
            if len(incident) == 1:
                normal = face_normals.get(incident[0])
                weight = self.boundary_weight

            elif len(incident) == 2:
                n1 = face_normals.get(incident[0])
                n2 = face_normals.get(incident[1])

                if n1 is None or n2 is None:
                    continue

                if np.dot(n1, n2) >= threshold:
                    continue

                normal = n1 + n2
                length = np.linalg.norm(normal)

                if length < 1e-12:
                    continue

                normal /= length
                weight = self.feature_weight

            else:
                continue

            if normal is None:
                continue

            direction = self.vertices[v] - self.vertices[u]
            length = np.linalg.norm(direction)

            if length < 1e-12:
                continue

            direction /= length
            constraint = np.cross(direction, normal)
            length = np.linalg.norm(constraint)

            if length < 1e-12:
                continue

            constraint /= length
            plane = np.append(
                constraint,
                -np.dot(constraint, self.vertices[u])
            )

            quadric = weight * np.outer(plane, plane)
            self.quadrics[u] += quadric
            self.quadrics[v] += quadric

    def build_topology(self):
        count = len(self.vertices)

        self.neighbors = [set() for _ in range(count)]
        self.incident_faces = [set() for _ in range(count)]
        self.edge_faces = {}

        for i, face in enumerate(self.faces):
            if self.active_faces[i]:
                self.add_face_topology(i, face)

    def add_face_topology(self, index, face):
        a, b, c = face.a, face.b, face.c

        for vertex in (a, b, c):
            self.incident_faces[vertex].add(index)

        for u, v in ((a, b), (b, c), (c, a)):
            edge = (min(u, v), max(u, v))
            self.neighbors[u].add(v)
            self.neighbors[v].add(u)
            self.edge_faces.setdefault(edge, []).append(index)

    def remove_face_topology(self, index, face):
        a, b, c = face.a, face.b, face.c

        for vertex in (a, b, c):
            self.incident_faces[vertex].discard(index)

        for u, v in ((a, b), (b, c), (c, a)):
            edge = (min(u, v), max(u, v))
            incident = self.edge_faces.get(edge)

            if incident is None:
                continue

            if index in incident:
                incident.remove(index)

            if not incident:
                del self.edge_faces[edge]
                self.neighbors[u].discard(v)
                self.neighbors[v].discard(u)

    def update_local_topology(self, old_faces, changed):
        for i, face in old_faces:
            self.remove_face_topology(i, face)

        for i, face in changed:
            self.add_face_topology(i, face)

    def is_valid_edge(self, u, v):
        edge = (min(u, v), max(u, v))
        incident = self.edge_faces.get(edge, [])

        if len(incident) not in (1, 2):
            return False

        common = self.neighbors[u] & self.neighbors[v]
        opposite = set()

        for i in incident:
            face = self.faces[i]
            opposite.update(
                x for x in (face.a, face.b, face.c)
                if x != u and x != v
            )

        if common != opposite:
            return False

        if len(incident) == 1:
            return True

        normals = []

        for i in incident:
            face = self.faces[i]
            a, b, c = (
                self.vertices[x]
                for x in (face.a, face.b, face.c)
            )

            normal = np.cross(b - a, c - a)
            length = np.linalg.norm(normal)

            if length < 1e-12:
                return False

            normals.append(normal / length)

        threshold = np.cos(np.radians(self.feature_angle))
        return np.dot(normals[0], normals[1]) >= threshold

    def get_candidate_positions(self, u, v):
        quadric = self.quadrics[u] + self.quadrics[v]
        p1, p2 = self.vertices[u], self.vertices[v]

        positions = []
        matrix = quadric[:3, :3]
        vector = -quadric[:3, 3]

        try:
            optimal = np.linalg.solve(matrix, vector)

            if np.all(np.isfinite(optimal)):
                positions.append(optimal)

        except np.linalg.LinAlgError:
            pass

        positions.extend((p1, p2, (p1 + p2) * 0.5))

        candidates = []
        seen = []

        for position in positions:
            position = np.asarray(position, dtype=float)

            if not np.all(np.isfinite(position)):
                continue

            if any(np.linalg.norm(position - p) < 1e-10 for p in seen):
                continue

            seen.append(position.copy())

            point = np.append(position, 1.0)
            cost = float(point @ quadric @ point)

            if np.isfinite(cost):
                candidates.append((max(0.0, cost), position.copy()))

        candidates.sort(key=lambda x: x[0])
        return candidates

    def check_geometry(self, u, v, position):
        faces = self.incident_faces[u] | self.incident_faces[v]

        for i in faces:
            face = self.faces[i]
            indices = (face.a, face.b, face.c)

            if u in indices and v in indices:
                continue

            old_points = [self.vertices[x] for x in indices]

            new_points = [
                position if x == u or x == v else self.vertices[x]
                for x in indices
            ]

            old_normal = np.cross(
                old_points[1] - old_points[0],
                old_points[2] - old_points[0]
            )

            new_normal = np.cross(
                new_points[1] - new_points[0],
                new_points[2] - new_points[0]
            )

            if np.linalg.norm(new_normal) < 1e-12:
                return False

            if np.dot(old_normal, new_normal) <= 0.0:
                return False

        return True

    def prepare_collapse(self, u, v):
        changed = []
        removed = []

        for i in self.incident_faces[u]:
            face = self.faces[i]

            vertices = [
                v if x == u else x
                for x in (face.a, face.b, face.c)
            ]

            if len(set(vertices)) < 3:
                removed.append(
                    (i, (face.a, face.b, face.c))
                )
            else:
                changed.append(
                    (i, obja.Face(*vertices))
                )

        return changed, removed

    def evaluate_collapse(self, u, v, candidates):
        if not self.is_valid_edge(u, v):
            return None

        changed, removed = self.prepare_collapse(u, v)

        if not removed:
            return None

        for cost, position in candidates:
            if self.check_geometry(u, v, position):
                return cost, position, changed, removed

        return None

    def invalidate_local_edges(self, vertices):
        edges = set()

        for vertex in vertices:
            for neighbor in self.neighbors[vertex]:
                edges.add(
                    (min(vertex, neighbor), max(vertex, neighbor))
                )

        for edge in edges:
            self.edge_versions[edge] = (
                self.edge_versions.get(edge, 0) + 1
            )

    def push_edge(self, u, v):
        edge = (min(u, v), max(u, v))

        if edge not in self.edge_faces:
            return

        candidates = self.get_candidate_positions(u, v)

        if not candidates:
            return

        version = self.edge_versions.get(edge, 0) + 1
        self.edge_versions[edge] = version

        heapq.heappush(
            self.heap,
            (
                candidates[0][0],
                self.heap_counter,
                u,
                v,
                version,
                candidates
            )
        )

        self.heap_counter += 1

    def build_heap(self):
        self.heap = []
        self.edge_versions = {}
        self.heap_counter = 0

        for u, v in self.edge_faces:
            self.push_edge(u, v)

    def apply_collapse(self, u, v, position, changed, removed):
        compact_changed = []

        for i, _ in changed:
            face = self.faces[i]
            corner = (face.a, face.b, face.c).index(u)
            compact_changed.append((i, corner))

        compact_removed = []
        removed_vertices = {}

        for i, vertices in removed:
            corner_u = vertices.index(u)
            corner_v = vertices.index(v)
            third = next(x for x in vertices if x != u and x != v)

            compact_removed.append(
                (i, vertices)
            )

            removed_vertices[third] = self.vertices[third].copy()

        operation = {
            "u": u,
            "v": v,
            "old_u_position": self.vertices[u].copy(),
            "old_v_position": self.vertices[v].copy(),
            "new_v_position": position.copy(),
            "changed_faces": compact_changed,
            "removed_faces": compact_removed,
            "removed_vertices": removed_vertices
        }

        self.operations.append(operation)

        self.vertices[v] = position.copy()
        self.quadrics[v] += self.quadrics[u]

        for i, face in changed:
            self.faces[i] = face.clone()

        for i, _ in removed:
            self.active_faces[i] = False

        return len(removed)

    def simplify_progressively(self, target_ratio=0.5, max_collapses=None):
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError(
                "target_ratio doit être compris entre 0 et 1."
            )

        self.initialize()

        initial_faces = sum(self.active_faces)

        if initial_faces == 0:
            raise ValueError(
                "Le maillage ne contient aucune face active."
            )

        target_faces = max(1, int(initial_faces * target_ratio))
        current_faces = initial_faces
        collapse_count = 0

        print("Faces initiales :", initial_faces)
        print("Faces ciblées :", target_faces)

        self.build_heap()

        while current_faces > target_faces:
            if (
                max_collapses is not None
                and collapse_count >= max_collapses
            ):
                break

            collapsed = False

            while self.heap:
                cost, _, u, v, version, candidates = heapq.heappop(
                    self.heap
                )

                edge = (min(u, v), max(u, v))

                if self.edge_versions.get(edge) != version:
                    continue

                if edge not in self.edge_faces:
                    continue

                result = self.evaluate_collapse(u, v, candidates)

                if result is None:
                    continue

                cost, position, changed, removed = result

                affected = (
                    self.neighbors[u]
                    | self.neighbors[v]
                    | {u, v}
                )

                old_faces = [
                    (i, self.faces[i].clone())
                    for i in self.incident_faces[u]
                ]

                self.invalidate_local_edges(affected)

                removed_count = self.apply_collapse(
                    u, v, position, changed, removed
                )

                self.update_local_topology(old_faces, changed)

                current_faces -= removed_count
                collapse_count += 1
                collapsed = True

                updated_edges = set()

                for vertex in affected:
                    for neighbor in self.neighbors[vertex]:
                        updated_edges.add(
                            (min(vertex, neighbor), max(vertex, neighbor))
                        )

                for a, b in updated_edges:
                    self.push_edge(a, b)

                print(
                    f"Collapse {collapse_count} : ({u}, {v}) "
                    f"| Erreur = {cost:.6e} "
                    f"| Faces = {current_faces}"
                )

                break

            if not collapsed:
                print("Aucune contraction valide restante.")
                break

        print("\nSimplification terminée.")
        print("Contractions :", collapse_count)
        print("Faces restantes :", current_faces)
        print(
            f"Ratio final : {current_faces / initial_faces:.3f}"
        )

    def write_progressive_mesh(self, output):
        writer = obja.Output(output, random_color=True)

        base_faces = [
            i for i, active in enumerate(self.active_faces)
            if active
        ]

        used_vertices = set()

        for i in base_faces:
            face = self.faces[i]
            used_vertices.update(
                (face.a, face.b, face.c)
            )

        for index in sorted(used_vertices):
            writer.add_vertex(index, self.vertices[index])

        for i in base_faces:
            writer.add_face(i, self.faces[i])

        for operation in reversed(self.operations):
            u = operation["u"]
            v = operation["v"]

            if u not in writer.vertex_mapping:
                writer.add_vertex(
                    u,
                    operation["old_u_position"]
                )

            if v not in writer.vertex_mapping:
                writer.add_vertex(
                    v,
                    operation["new_v_position"]
                )

            writer.edit_vertex(
                v,
                operation["old_v_position"]
            )

            for i, corner in operation["changed_faces"]:
                print(
                    f"efv {writer.face_mapping[i] + 1} "
                    f"{corner + 1} "
                    f"{writer.vertex_mapping[u] + 1}",
                    file=output
                )

            for i, vertices in operation["removed_faces"]:
                for vertex in vertices:
                    if vertex not in writer.vertex_mapping:
                        if vertex == u:
                            position = operation["old_u_position"]
                        elif vertex == v:
                            position = operation["old_v_position"]
                        else:
                            position = operation["removed_vertices"][vertex]

                        writer.add_vertex(vertex, position)

                writer.add_face(
                    i,
                    obja.Face(*vertices)
                )

        print("Fichier progressif écrit.")


def main():
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = "example/suzanne.obj"
        output_file = "example/suzanne_pm.obja"

    model = ProgressiveMesh()

    print("Chargement de", input_file)
    model.parse_file(input_file)

    print("Sommets :", len(model.vertices))
    print("Faces :", len(model.faces))

    model.simplify_progressively(target_ratio=0.13)

    with open(output_file, "w") as output:
        model.write_progressive_mesh(output)

    print("Maillage progressif généré :", output_file)


if __name__ == "__main__":
    main()
