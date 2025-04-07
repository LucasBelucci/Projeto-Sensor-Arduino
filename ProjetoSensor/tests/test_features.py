import numpy as np
import sys
from pathlib import Path
from ProjetoSensor.esp32.analysis import extract_fft_features
from esp32.analysis import extract_fft_features
from esp32.training import mahalonobis_distance

#sys.path.append(str(Path(__file__).resolve().parent.parent))

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