import numpy as np
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch
import tempfile
import os

# Caminho para o diretório raiz do projeto
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from esp32.analysis import extract_fft_features, analyze_statistics, plot_fft_comparison
from esp32.training import mahalonobis_distance

def test_extract_fft_features_shape():
    sample = np.random.randn(100, 3)
    features = extract_fft_features(sample)
    
    assert features.shape == (50, 3)

def test_mahalonobis_distance():
    x = np.random.randn(10, 3)
    mean = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    distances = mahalonobis_distance(x, mean, cov_inv)

    assert len(distances) == 10
    assert all(d >= 0 for d in distances)

@patch("esp32.analysis.load_samples")
def test_analyze_statistics(mock_load_sample):
    fake_data = np.random.randn(100, 3)
    mock_load_sample.return_value = fake_data

    result = analyze_statistics("dummy.csv")

    assert "Mean" in result
    assert result["Mean"].shape == (3,)
    assert result["Variance"].shape == (3,)
    assert result["Kurtosis"].shape == (3, )
    assert result["Skew"].shape == (3,)
    assert result["MAD"].shape == (3,)
    assert result["Correlation"].shape == (3, 3)
    


@patch("esp32.analysis.extract_fft_features")
@patch("esp32.analysis.load_samples")
def test_plot_fft_comparison(mock_load_sample, mock_extract_fft):
    fake_sample = np.random.randn(100, 3)
    normal_files = ["normal.csv", "normal_2.csv"]
    anomaly_files = ["anomaly.csv", "anomaly_2.csv"]

    
    mock_load_sample.return_value = fake_sample

    fake_fft = np.abs(np.random.randn(50, 3)) + 0.1
    mock_extract_fft.return_value = fake_fft
    try: 
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    
            fig = plot_fft_comparison(normal_files, anomaly_files, num_samples=2, save_path=tmp_file.name)
            assert os.path.exists(tmp_file.name)
            assert os.path.getsize(tmp_file.name) > 0
    finally:
            os.remove(tmp_file.name)
    
    #fig = plot_fft_comparison(normal_files, anomaly_files, num_samples=2)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3
    assert fig.axes[0].get_title() != ""

    plt.close(fig)
    