"""Check the reference subtraction without loading model checkpoints."""
import ast
from pathlib import Path
import unittest

import numpy as np


class Matrix:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float64)

    def mean(self, dim, keepdim=False):
        return Matrix(self.value.mean(axis=dim, keepdims=keepdim))

    def __sub__(self, other):
        return Matrix(self.value - other.value)


class ReferenceAssignmentTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parent / "code/exp6_draft_intra4.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "layout_reference_scores"]
        self.assertEqual(len(functions), 1, "The layout-reference assignment function is missing")
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        self.score = namespace["layout_reference_scores"]

    def test_pair_preference_removes_the_reference_pair_preference(self):
        baseline = np.array([[5., 1.], [2., 4.]])
        measured = np.array([[8., 1.], [2., 7.]])
        adjusted = self.score(Matrix(measured), Matrix(baseline)).value
        exchange = lambda m: m[0, 0] + m[1, 1] - m[0, 1] - m[1, 0]
        self.assertAlmostEqual(exchange(adjusted), exchange(measured) - exchange(baseline))

    def test_reference_only_row_or_column_constants_cancel(self):
        scores = np.arange(16, dtype=np.float64).reshape(4, 4)
        reference = np.arange(4)[:, None] + np.array([2, 8, 3, 1])[None, :]
        adjusted = self.score(Matrix(scores), Matrix(reference)).value
        np.testing.assert_allclose(adjusted, scores)

    def test_absent_reference_preserves_control_tensor(self):
        scores = Matrix([[0., 2.], [3., 1.]])
        self.assertIs(self.score(scores, None), scores)


if __name__ == "__main__":
    unittest.main()
