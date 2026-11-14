import io
import unittest

import numpy as np

from progressive_mesh import ProgressiveMesh


class ChenNishitaTests(unittest.TestCase):
    def test_chen_simplification_uses_half_edge_without_ev_and_reverses(self):
        model = ProgressiveMesh(use_chen=True)
        model.parse_file("example/suzanne.obj")
        initial_positions = np.asarray(model.vertices, dtype=float).copy()
        initial_faces = model.count_visible_faces()

        model.simplify(target_ratio=0.1)

        self.assertLessEqual(model.count_visible_faces(), max(1, int(initial_faces * 0.1)))
        self.assertGreater(len(model.history), 0)

        # En Half-Edge Collapse, le sommet conservé vt ne doit jamais se déplacer
        for collapse in model.history:
            np.testing.assert_allclose(
                model.vertices[collapse.target_vertex_id],
                collapse.target_vertex_position,
            )

        # L'export OBJA ne doit donc émettre aucune instruction 'ev'
        buf = io.StringIO()
        model.write_obja(buf)
        self.assertEqual(model._last_obja_stats["ev_written"], 0)

        # Tous les collapses doivent être inversibles via vertex_split
        while model.history:
            self.assertTrue(model.vertex_split())
        self.assertEqual(model.count_visible_faces(), initial_faces)
        np.testing.assert_allclose(np.asarray(model.vertices, dtype=float), initial_positions)


if __name__ == "__main__":
    unittest.main()
