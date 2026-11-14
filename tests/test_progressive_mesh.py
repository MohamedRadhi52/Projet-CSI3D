import os
import tempfile
import unittest

import numpy as np

import obja
from comparaison import compare_models
from progressive_mesh import ProgressiveMesh


class ProgressiveMeshTests(unittest.TestCase):
    def setUp(self):
        self.model = ProgressiveMesh()
        self.model.parse_file("example/suzanne.obj")

    def test_loads_suzanne_and_builds_topology(self):
        self.assertEqual(len(self.model.vertices), 511)
        self.assertEqual(self.model.count_visible_faces(), 968)

        edges = self.model.collect_edges()

        self.assertGreater(len(edges), 0)
        self.assertEqual(len(self.model.edge_faces), len(edges))
        self.assertTrue(self.model.vertex_neighbors)

    def test_total_hoppe_energy_combines_qem_spring_and_representation(self):
        self.model.collect_edges()
        self.model._rebuild_quadrics()
        edge = next(iter(self.model.edge_faces))
        candidate, total = self.model._candidate_position(*edge)

        expected = (
            self.model._qem_cost(*edge, candidate)
            + self.model.lambda_spring
            * self.model._spring_energy(*edge, candidate)
            + self.model.lambda_rep
        )

        self.assertAlmostEqual(total, expected)
        self.assertAlmostEqual(
            self.model.edge_length(*edge),
            expected
        )

    def test_simplification_records_collapses(self):
        initial_faces = self.model.count_visible_faces()

        self.model.simplify(target_ratio=0.98)

        self.assertLess(self.model.count_visible_faces(), initial_faces)
        self.assertGreater(len(self.model.history), 0)
        self.assertEqual(
            self.model.count_visible_faces(),
            948
        )

    def test_vertex_split_reverses_last_collapse(self):
        self.model.simplify(target_ratio=0.98)
        simplified_faces = self.model.count_visible_faces()
        history_size = len(self.model.history)

        self.assertTrue(self.model.vertex_split())

        self.assertGreater(self.model.count_visible_faces(), simplified_faces)
        self.assertEqual(len(self.model.history), history_size - 1)
        with self.assertRaises(ValueError):
            self.model.vertex_split(object())

    def test_quantization_preserves_shape_and_rejects_invalid_bits(self):
        positions = np.asarray(self.model.vertices)
        quantized, minimum, maximum = ProgressiveMesh._quantize_positions(
            positions,
            bits=16
        )
        reconstructed = ProgressiveMesh.dequantize_vertices(
            quantized,
            (minimum, maximum),
            bits=16
        )

        self.assertEqual(reconstructed.shape, positions.shape)
        self.assertLess(
            float(np.max(np.abs(positions - reconstructed))),
            0.001
        )
        with self.assertRaises(ValueError):
            ProgressiveMesh._quantize_positions(positions, bits=0)

    def test_progressive_binary_round_trip(self):
        self.model.simplify(target_ratio=0.98)
        expected_faces = self.model.count_visible_faces()
        expected_history = len(self.model.history)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pmz")
            size = self.model.write_progressive_binary(path, bits=16)
            restored = ProgressiveMesh.read_progressive_binary(path)

        self.assertGreater(size, 44)
        self.assertEqual(restored.count_visible_faces(), expected_faces)
        self.assertEqual(len(restored.history), expected_history)
        self.assertTrue(restored.vertex_split())

    def test_corrupted_progressive_binary_is_rejected(self):
        with self.assertRaises(ValueError):
            ProgressiveMesh._decompress_progressive_mesh(b"invalid")

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "corrupted.pmz")
            with open(path, "wb") as file:
                file.write(b"PMZ1" + b"\x00" * 40)

            with self.assertRaises(ValueError):
                ProgressiveMesh.read_progressive_binary(path)

    def test_comparison_detects_identical_model(self):
        original = obja.parse_file("example/suzanne.obj")
        result = compare_models(original, original)

        self.assertEqual(result["original_vertices"], 511)
        self.assertEqual(result["final_vertices"], 511)
        self.assertEqual(result["original_faces"], 968)
        self.assertEqual(result["final_faces"], 968)
        self.assertEqual(result["maximum_geometric_error"], 0.0)

    def test_robustness_on_multiple_models(self):
        models = {
            "tetraedre": (
                "v 0 0 0\n"
                "v 1 0 0\n"
                "v 0.5 1 0\n"
                "v 0.5 0.5 1\n"
                "f 1 3 2\n"
                "f 1 2 4\n"
                "f 2 3 4\n"
                "f 3 1 4\n"
            ),
            "triangle": (
                "v 0 0 0\n"
                "v 1 0 0\n"
                "v 0 1 0\n"
                "f 1 2 3\n"
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            for name, content in models.items():
                with self.subTest(model=name):
                    source_path = os.path.join(directory, name + ".obj")
                    archive_path = os.path.join(directory, name + ".pmz")
                    with open(source_path, "w", encoding="utf-8") as file:
                        file.write(content)

                    model = ProgressiveMesh()
                    model.parse_file(source_path)
                    initial_faces = model.count_visible_faces()
                    model.collect_edges()
                    model.simplify(target_ratio=0.5)
                    model.write_progressive_binary(archive_path, bits=12)
                    restored = ProgressiveMesh.read_progressive_binary(
                        archive_path
                    )

                    self.assertGreater(initial_faces, 0)
                    self.assertGreater(len(model.vertices), 0)
                    self.assertLessEqual(
                        restored.count_visible_faces(),
                        initial_faces
                    )
                    self.assertEqual(
                        len(restored.history),
                        len(model.history)
                    )


if __name__ == "__main__":
    unittest.main()
