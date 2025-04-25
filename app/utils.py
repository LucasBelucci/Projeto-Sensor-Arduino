import numpy as np

from ProjetoSensor.esp32.analysis import extract_fft_features


# Distância de Mahalonobis
def mahalonobis_distance(x, mean, cov_inv):
    x = np.atleast_2d(x)
    mean = np.atleast_2d(mean)
    diff = x - mean
    dist = np.sqrt(np.sum(np.dot(diff, cov_inv) * diff, axis=1))
    print("Distância de Mahalanobis:", dist)
    return dist

#def adjust_threshold(distances, percentile=95):
#    threshold = np.percentile(distances, percentile)
#    return threshold              

# Verifica a anomalia
def check_anomaly(features, mean, cov_inv, threshold):
    #features = extract_fft_features(sample)
    dist = mahalonobis_distance(features, mean, cov_inv)
    return dist, dist > threshold