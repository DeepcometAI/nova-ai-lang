"""
Unittest runner for Step 12 stdlib checks.

This is a pytest-free mirror of `test_stdlib_step12.py`, useful on machines
where pytest isn't installed yet.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

# Add stdlib to path (same as the pytest integration test)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "stdlib"))


class TestCosmosStats(unittest.TestCase):
    def test_pearson_basic(self):
        from cosmos.stats import pearson

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        r = pearson(x, y)
        self.assertLess(abs(r - 1.0), 1e-10)

    def test_pearson_negative(self):
        from cosmos.stats import pearson

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 8, 6, 4, 2])
        r = pearson(x, y)
        self.assertLess(abs(r - (-1.0)), 1e-10)

    def test_spearman_basic(self):
        from cosmos.stats import spearman

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        rho = spearman(x, y)
        self.assertLess(abs(rho - 1.0), 1e-10)

    def test_linear_fit_basic(self):
        from cosmos.stats import linear_fit

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        slope, intercept = linear_fit(x, y)
        self.assertLess(abs(slope - 2.0), 1e-10)
        self.assertLess(abs(intercept - 0.0), 1e-10)

    def test_mean_median_std_min_max_sum_polyfit(self):
        from cosmos.stats import mean, median, std, min, max, sum, polyfit

        x = np.array([1, 2, 3, 4, 5])
        self.assertLess(abs(mean(x) - 3.0), 1e-10)
        self.assertLess(abs(median(x) - 3.0), 1e-10)
        expected_std = np.std([1, 2, 3, 4, 5])
        self.assertLess(abs(std(x) - expected_std), 1e-10)

        y = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertEqual(min(y), 1.0)
        self.assertEqual(max(y), 9.0)
        self.assertEqual(sum(x), 15.0)

        # polyfit degree 2 should yield 3 coefficients
        coeffs = polyfit(np.array([1, 2, 3]), np.array([1, 4, 9]), 2)
        self.assertEqual(len(coeffs), 3)


class TestCosmosData(unittest.TestCase):
    def test_csv_roundtrip(self):
        from cosmos.data import write_csv, read_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            data = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
            write_csv(data, path)

            wave = read_csv(path)
            collected = wave.collect()
            self.assertEqual(len(collected), 2)
            self.assertEqual(collected[0]["x"], 1)

    def test_wave_filter_map_batch(self):
        from cosmos.data import Wave

        def gen():
            for i in range(10):
                yield i

        wave = Wave(gen())
        self.assertEqual(wave.filter(lambda x: x > 2).collect()[:2], [3, 4])
        self.assertEqual(wave.map(lambda x: x * 2).collect()[:5], [0, 2, 4, 6, 8])
        batches = Wave(gen()).batch(3).collect()
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0], [0, 1, 2])


class TestCosmosML(unittest.TestCase):
    def test_mse_relu_sigmoid_softmax_tanh_gelu_linear_batch_norm(self):
        from cosmos.ml import mse, relu, sigmoid, softmax, tanh, gelu, linear, batch_norm

        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 2.1, 2.9])
        loss = mse(pred, target)
        self.assertLess(abs(loss - np.mean([0.01, 0.01, 0.01])), 1e-10)

        y = relu(np.array([-1, 0, 1, 2]))
        self.assertTrue(np.allclose(y, np.array([0, 0, 1, 2])))

        self.assertLess(abs(sigmoid(np.array([0.0]))[0] - 0.5), 1e-10)

        sm = softmax(np.array([[1.0, 2.0, 3.0]]))
        self.assertLess(abs(np.sum(sm) - 1.0), 1e-10)
        self.assertGreater(sm[0, 2], sm[0, 1])
        self.assertGreater(sm[0, 1], sm[0, 0])

        self.assertLess(abs(tanh(np.array([0.0]))[0] - 0.0), 1e-10)
        self.assertLess(abs(gelu(np.array([0.0]))[0] - 0.0), 1e-10)

        w = linear(10, 5)
        self.assertEqual(w.shape, (5, 10))

        bn = batch_norm(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertLess(abs(np.mean(bn)), 1e-10)
        # Loose std tolerance (matches pytest test intent)
        self.assertLess(abs(np.std(bn) - 1.0), 1e-1)


class TestCosmosPlot(unittest.TestCase):
    def test_scatter_plot_set_title(self):
        from cosmos.plot import scatter, plot, set_title, _ensure_fig
        import matplotlib.pyplot as plt

        scatter(np.array([1, 2, 3]), np.array([1, 2, 3]))
        fig, ax = _ensure_fig()
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        plt.close("all")

        plot(np.array([1, 2, 3]), np.array([1, 2, 3]))
        fig, ax = _ensure_fig()
        self.assertGreater(len(ax.lines), 0)
        plt.close("all")

        set_title("Test Title")
        fig, ax = _ensure_fig()
        self.assertEqual(ax.get_title(), "Test Title")
        plt.close("all")


class TestNovaFS(unittest.TestCase):
    def test_write_read_exists_list_mkdir_copy(self):
        from nova.fs import write_file, read_file, file_exists, list_dir, mkdir, is_dir, copy_file

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            write_file(path, "Hello, world!")
            self.assertEqual(read_file(path), "Hello, world!")
            self.assertTrue(file_exists(path))

            write_file(os.path.join(tmpdir, "a.txt"), "a")
            files = list_dir(tmpdir)
            self.assertIn("a.txt", files)

            subdir = os.path.join(tmpdir, "subdir")
            ok, err = mkdir(subdir)
            self.assertTrue(ok, err)
            self.assertTrue(is_dir(subdir))

            src = os.path.join(tmpdir, "src.txt")
            dst = os.path.join(tmpdir, "dst.txt")
            write_file(src, "source content")
            ok, err = copy_file(src, dst)
            self.assertTrue(ok, err)
            self.assertEqual(read_file(dst), "source content")


class TestNovaCLI(unittest.TestCase):
    def test_add_argument_and_input_optional_confirm(self):
        from nova.cli import add_argument, _get_parser, input_optional, confirm
        from unittest import mock

        parser = _get_parser()
        add_argument("test_arg", type_name="str", required=False, help_text="Test")
        args = parser.parse_args(["--test_arg", "value"])
        self.assertEqual(args.test_arg, "value")

        with mock.patch("builtins.input", return_value=""):
            result = input_optional("Enter value: ", default="default_value")
            self.assertEqual(result, "default_value")

        with mock.patch("builtins.input", return_value="y"):
            self.assertTrue(confirm("Continue?"))
        with mock.patch("builtins.input", return_value="n"):
            self.assertFalse(confirm("Continue?"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

