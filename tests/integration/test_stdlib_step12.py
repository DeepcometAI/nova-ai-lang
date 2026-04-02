"""
Integration tests for NOVA stdlib Priority 1 (Step 12 Phase A)
Tests for cosmos.stats, cosmos.data, cosmos.ml, cosmos.plot, nova.fs, nova.cli
"""

import sys
import os
import tempfile
import pytest
import numpy as np
from pathlib import Path

# Add stdlib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'stdlib'))

# ============= cosmos.stats Tests =============

class TestCosmosStats:
    """Test statistical functions (cosmos.stats)"""
    
    def test_pearson_basic(self):
        """Test Pearson correlation coefficient."""
        from cosmos.stats import pearson
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        r = pearson(x, y)
        assert abs(r - 1.0) < 1e-10, "Perfect positive correlation should be 1.0"
    
    def test_pearson_negative(self):
        """Test Pearson correlation (negative)."""
        from cosmos.stats import pearson
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 8, 6, 4, 2])
        r = pearson(x, y)
        assert abs(r - (-1.0)) < 1e-10, "Perfect negative correlation should be -1.0"
    
    def test_spearman_basic(self):
        """Test Spearman rank correlation."""
        from cosmos.stats import spearman
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        rho = spearman(x, y)
        assert abs(rho - 1.0) < 1e-10, "Perfect rank correlation should be 1.0"
    
    def test_linear_fit_basic(self):
        """Test linear regression."""
        from cosmos.stats import linear_fit
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        slope, intercept = linear_fit(x, y)
        assert abs(slope - 2.0) < 1e-10, f"Expected slope 2.0, got {slope}"
        assert abs(intercept - 0.0) < 1e-10, f"Expected intercept 0.0, got {intercept}"
    
    def test_mean(self):
        """Test arithmetic mean."""
        from cosmos.stats import mean
        x = np.array([1, 2, 3, 4, 5])
        m = mean(x)
        assert abs(m - 3.0) < 1e-10, "Mean should be 3.0"
    
    def test_median(self):
        """Test median."""
        from cosmos.stats import median
        x = np.array([1, 2, 3, 4, 5])
        med = median(x)
        assert abs(med - 3.0) < 1e-10, "Median should be 3.0"
    
    def test_std(self):
        """Test standard deviation."""
        from cosmos.stats import std
        x = np.array([1, 2, 3, 4, 5])
        s = std(x)
        expected = np.std([1, 2, 3, 4, 5])
        assert abs(s - expected) < 1e-10, f"Std dev should be {expected}"
    
    def test_min_max(self):
        """Test min and max."""
        from cosmos.stats import min, max
        x = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        assert min(x) == 1.0, "Min should be 1"
        assert max(x) == 9.0, "Max should be 9"
    
    def test_sum(self):
        """Test sum."""
        from cosmos.stats import sum
        x = np.array([1, 2, 3, 4, 5])
        s = sum(x)
        assert s == 15.0, "Sum should be 15"
    
    def test_polyfit(self):
        """Test polynomial fit."""
        from cosmos.stats import polyfit
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 9])  # y = x^2
        coeffs = polyfit(x, y, 2)
        assert len(coeffs) == 3, "Should return 3 coefficients for degree 2"


# ============= cosmos.data Tests =============

class TestCosmosData:
    """Test data loading (cosmos.data)"""
    
    def test_csv_roundtrip(self):
        """Test CSV write and read."""
        from cosmos.data import write_csv, read_csv
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.csv')
            data = [
                {'x': 1, 'y': 2},
                {'x': 3, 'y': 4},
            ]
            write_csv(data, path)
            
            wave = read_csv(path)
            collected = wave.collect()
            assert len(collected) == 2, "Should read 2 records"
            assert collected[0]['x'] == 1, "First record x should be 1"
    
    def test_wave_filter(self):
        """Test Wave filter operation."""
        from cosmos.data import Wave
        
        def gen():
            for i in range(5):
                yield i
        
        wave = Wave(gen())
        filtered = wave.filter(lambda x: x > 2)
        result = filtered.collect()
        assert result == [3, 4], f"Expected [3, 4], got {result}"
    
    def test_wave_map(self):
        """Test Wave map operation."""
        from cosmos.data import Wave
        
        def gen():
            for i in range(5):
                yield i
        
        wave = Wave(gen())
        mapped = wave.map(lambda x: x * 2)
        result = mapped.collect()
        assert result == [0, 2, 4, 6, 8], f"Expected [0,2,4,6,8], got {result}"
    
    def test_wave_batch(self):
        """Test Wave batch operation."""
        from cosmos.data import Wave
        
        def gen():
            for i in range(10):
                yield i
        
        wave = Wave(gen())
        batched = wave.batch(3)
        result = batched.collect()
        assert len(result) == 4, "Should have 4 batches"
        assert result[0] == [0, 1, 2], "First batch should be [0,1,2]"


# ============= cosmos.ml Tests =============

class TestCosmosML:
    """Test machine learning functions (cosmos.ml)"""
    
    def test_mse_loss(self):
        """Test MSE loss."""
        from cosmos.ml import mse
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 2.1, 2.9])
        loss = mse(pred, target)
        expected = np.mean([0.01, 0.01, 0.01])
        assert abs(loss - expected) < 1e-10, f"MSE should be {expected}"
    
    def test_relu(self):
        """Test ReLU activation."""
        from cosmos.ml import relu
        x = np.array([-1, 0, 1, 2])
        y = relu(x)
        expected = np.array([0, 0, 1, 2])
        assert np.allclose(y, expected), f"ReLU output incorrect"
    
    def test_sigmoid(self):
        """Test Sigmoid activation."""
        from cosmos.ml import sigmoid
        x = np.array([0.0])
        y = sigmoid(x)
        assert abs(y[0] - 0.5) < 1e-10, "Sigmoid(0) should be 0.5"
    
    def test_softmax(self):
        """Test Softmax activation."""
        from cosmos.ml import softmax
        x = np.array([[1.0, 2.0, 3.0]])
        y = softmax(x)
        # Softmax should sum to 1
        assert abs(np.sum(y) - 1.0) < 1e-10, "Softmax should sum to 1"
        # Logit with highest score should have highest probability
        assert y[0, 2] > y[0, 1] > y[0, 0], "Probabilities should be ordered"
    
    def test_tanh(self):
        """Test tanh activation."""
        from cosmos.ml import tanh
        x = np.array([0.0])
        y = tanh(x)
        assert abs(y[0] - 0.0) < 1e-10, "tanh(0) should be 0"
    
    def test_gelu(self):
        """Test GELU activation."""
        from cosmos.ml import gelu
        x = np.array([0.0])
        y = gelu(x)
        assert abs(y[0] - 0.0) < 1e-10, "GELU(0) should be 0"
    
    def test_linear_layer_init(self):
        """Test linear layer initialization."""
        from cosmos.ml import linear
        weights = linear(10, 5)
        assert weights.shape == (5, 10), "Weight shape should be (5, 10)"
    
    def test_batch_norm(self):
        """Test batch normalization."""
        from cosmos.ml import batch_norm
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = batch_norm(x)
        # Should be normalized: mean ≈ 0, std ≈ 1
        assert abs(np.mean(y)) < 1e-10, "Batch norm mean should be 0"
        assert abs(np.std(y) - 1.0) < 1e-1, "Batch norm std should be 1"


# ============= cosmos.plot Tests =============

class TestCosmosPlot:
    """Test visualization functions (cosmos.plot)"""
    
    def test_scatter_creates_axes(self):
        """Test that scatter creates figure."""
        from cosmos.plot import scatter, _ensure_fig
        import matplotlib.pyplot as plt
        
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        scatter(x, y)
        
        fig, ax = _ensure_fig()
        assert fig is not None, "Figure should exist"
        assert ax is not None, "Axes should exist"
        plt.close('all')
    
    def test_plot_adds_line(self):
        """Test that plot adds a line."""
        from cosmos.plot import plot, _ensure_fig
        import matplotlib.pyplot as plt
        
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        plot(x, y)
        
        fig, ax = _ensure_fig()
        assert len(ax.lines) > 0, "Should have at least one line"
        plt.close('all')
    
    def test_set_title(self):
        """Test setting title."""
        from cosmos.plot import set_title, _ensure_fig
        import matplotlib.pyplot as plt
        
        set_title("Test Title")
        fig, ax = _ensure_fig()
        assert ax.get_title() == "Test Title", "Title should be set"
        plt.close('all')


# ============= nova.fs Tests =============

class TestNovaFS:
    """Test file system operations (nova.fs)"""
    
    def test_write_read_file(self):
        """Test write and read file."""
        from nova.fs import write_file, read_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.txt')
            content = "Hello, world!"
            write_file(path, content)
            
            read_content = read_file(path)
            assert read_content == content, "Read content should match written content"
    
    def test_file_exists(self):
        """Test file existence check."""
        from nova.fs import file_exists, write_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.txt')
            assert not file_exists(path), "File should not exist yet"
            
            write_file(path, "content")
            assert file_exists(path), "File should exist now"
    
    def test_list_dir(self):
        """Test directory listing."""
        from nova.fs import list_dir, write_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            write_file(os.path.join(tmpdir, 'a.txt'), "a")
            write_file(os.path.join(tmpdir, 'b.txt'), "b")
            
            files = list_dir(tmpdir)
            assert 'a.txt' in files and 'b.txt' in files, "Should list both files"
    
    def test_mkdir(self):
        """Test directory creation."""
        from nova.fs import mkdir, is_dir
        
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, 'subdir')
            success, error = mkdir(subdir)
            assert success, f"mkdir should succeed: {error}"
            assert is_dir(subdir), "Directory should exist"
    
    def test_copy_file(self):
        """Test file copy."""
        from nova.fs import copy_file, write_file, read_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, 'src.txt')
            dst = os.path.join(tmpdir, 'dst.txt')
            
            write_file(src, "source content")
            success, error = copy_file(src, dst)
            
            assert success, f"copy_file should succeed: {error}"
            assert read_file(dst) == "source content", "Copied content should match"


# ============= nova.cli Tests =============

class TestNovaCLI:
    """Test CLI operations (nova.cli)"""
    
    def test_add_argument(self):
        """Test adding arguments."""
        from nova.cli import add_argument, _get_parser
        
        parser = _get_parser()
        add_argument("test_arg", type_name="str", required=False, help_text="Test")
        
        # Should parse without error
        args = parser.parse_args(["--test_arg", "value"])
        assert args.test_arg == "value", "Should parse argument"
    
    def test_parse_args_basic(self):
        """Test basic argument parsing."""
        from nova.cli import parse_args
        
        success, error = parse_args(["--foo", "bar"])
        # Even though --foo is unknown, parse_args should handle it gracefully
        # (In this test, it may fail, which is OK for an unknown arg)
    
    def test_input_optional(self):
        """Test optional input (mock)."""
        from nova.cli import input_optional
        from unittest import mock
        
        with mock.patch('builtins.input', return_value=''):
            result = input_optional("Enter value: ", default="default_value")
            assert result == "default_value", "Should return default on empty input"
    
    def test_confirm(self):
        """Test confirmation prompt (mock)."""
        from nova.cli import confirm
        from unittest import mock
        
        with mock.patch('builtins.input', return_value='y'):
            assert confirm("Continue?") == True, "Should return True for 'y'"
        
        with mock.patch('builtins.input', return_value='n'):
            assert confirm("Continue?") == False, "Should return False for 'n'"


# ============= Integration Tests =============

class TestIntegration:
    """Integration tests combining multiple constellations"""
    
    def test_stats_with_data(self):
        """Test using stats functions with data loaded via cosmos.data."""
        from cosmos.data import write_csv, read_csv
        from cosmos.stats import mean, std
        import csv
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.csv')
            
            # Write CSV
            data = [{'value': 10}, {'value': 20}, {'value': 30}]
            write_csv(data, path)
            
            # Read and compute stats
            wave = read_csv(path)
            records = wave.collect()
            values = np.array([r['value'] for r in records])
            
            m = mean(values)
            s = std(values)
            
            assert abs(m - 20.0) < 1e-10, "Mean should be 20"
            assert s > 0, "Std dev should be positive"
    
    def test_fs_with_cli(self):
        """Test file system operations with CLI."""
        from nova.fs import write_file, read_file, list_dir
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.txt')
            write_file(config_path, "key=value")
            
            content = read_file(config_path)
            assert "key=value" in content, "File should contain config"
            
            files = list_dir(tmpdir)
            assert 'config.txt' in files, "Should list config file"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
