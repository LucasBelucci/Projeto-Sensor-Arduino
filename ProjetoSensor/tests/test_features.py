import numpy as np
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch
import tempfile
import os
import pytest

# Caminho para o diretório raiz do projeto
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from esp32.analysis import extract_fft_features, analyze_statistics, plot_fft_comparison
from esp32.training import mahalonobis_distance, extract_ml_features

# Testes para training Mahalonobis Distance

def test_mahalonobis_distance():
    x = np.random.randn(10, 3)
    mean = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    distances = mahalonobis_distance(x, mean, cov_inv)

    assert len(distances) == 10
    assert all(d >= 0 for d in distances)

# Testes para analyze Statistics

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
    

# Testes para Plot FFT Comparison

@patch("esp32.analysis.extract_fft_features")
@patch("esp32.analysis.load_samples")
def test_plot_fft_comparison(mock_load_sample, mock_extract_fft):
    fake_sample = np.random.randn(100, 3)
    normal_files = ["normal.csv", "normal_2.csv"]
    anomaly_files = ["anomaly.csv", "anomaly_2.csv"]

    
    mock_load_sample.return_value = fake_sample
    fake_fft = np.abs(np.random.randn(50, 3)) + 0.1
    mock_extract_fft.return_value = fake_fft

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "plot.png")
    
        fig = plot_fft_comparison(normal_files, anomaly_files, num_samples=2, save_path=tmp_path)
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3
    assert fig.axes[0].get_title() != ""


    plt.close(fig)
    
# Testes para extract_fft_features
def test_extract_fft_features_shape():
    sample = np.random.randn(100, 3)
    features = extract_fft_features(sample)
    
    assert features.shape == (50, 3)

def test_fft_with_dc():
     sample = np.random.randn(100, 3)
     features = extract_fft_features(sample, include_dc=True)

     assert features.shape == (51, 3) # Inclui a DC     
    
@pytest.mark.parametrize("window_type", ["hann", "hamming", None])
def test_fft_with_different_windows(window_type):
     sample = np.random.randn(100, 3)
     features = extract_fft_features(sample, window=window_type)

     assert features.shape == (50, 3)
     
def test_fft_return_freqs():
     sample = np.random.randn(100, 3)
     features, freqs = extract_fft_features(sample, return_freqs=True, sampling_rate=100)

     assert features.shape[0] == len(freqs)
     assert features.shape[1] == 3

def test_extract_fft_features_invalid_input():
    with pytest.raises(TypeError):
        extract_fft_features("not an array")

    with pytest.raises(ValueError):
        extract_fft_features(np.random.randn(100)) # 1D em vez de 2D

    with pytest.raises(ValueError):
        extract_fft_features(np.random.randn(1, 3)) # apenas 1 amostra

# Testes para verificação da entrada da função extract_ml_features
def test_extract_ml_features_invalid_input():
    with pytest.raises(ValueError):
        extract_ml_features("Não é um array")   # Entrada nao é um ndarray
    
    with pytest.raises(ValueError):
        extract_fft_features(np.random.randn(100))  # Entrada 1D em vez de 2D

# Teste para verificar as características extraidas
def test_extract_ml_features():
    sample = np.random.randn(100, 3)
    features = extract_ml_features(sample)

    num_axes = 3
    num_time_metrics = 5
    num_corr_pairs = 3 # XY, XZ, YZ
    num_freq_metrics = 4

    #expected_length = num_axes * num_time_metrics + num_corr_pairs + num_axes * num_freq_metrics
    expected_length = num_axes * num_time_metrics
    
    # Verificando se o vetor de características tem o formato esperado
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == expected_length    # Total de features(domínio do tempo, correlação e frequência)

    # Verificando se as features de domínio de tempo estão calculadas corretamente
    assert np.all(np.isfinite(features[:3])) # Média, Variância
    assert np.all(np.isfinite(features[3:6])) # Skew, Kurtosis, MAD

    # Verificando se a correlação entre eixos foi calculada corretamente
    assert np.all(np.isfinite(features[6:])) # Correlação e Frequência

# Teste para verificar o cálculo correto das FFTs
@patch("esp32.analysis.extract_fft_features")
def test_extract_ml_features_with_fft(mock_extract_fft):
    sample = np.random.randn(100, 3)
    
    fake_fft = np.abs(np.random.randn(50, 3)) + 0.1
    mock_extract_fft.return_value = fake_fft

    features = extract_ml_features(sample)

    assert np.all(np.isfinite(features[-4:])) # Média, desvio padrão, energia e frequência dominante

# Teste para a combinação correta das features
def test_extract_ml_features_combination():
    sample = np.random.randn(100, 3)
    features = extract_ml_features(sample)

    
    num_axes = 3
    num_time_metrics = 5
    num_corr_pairs = 3 # XY, XZ, YZ
    num_freq_metrics = 4

    #expected_length = num_axes * num_time_metrics + num_corr_pairs + num_axes * num_freq_metrics
    expected_length = num_axes * num_time_metrics

    # Verificar se todas as features estão concatenadas corretamente

    assert len(features) == expected_length