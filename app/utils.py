import numpy as np

from ProjetoSensor.esp32.analysis import extract_fft_features


# Distância de Mahalonobis
def mahalonobis_distance(x, mu, cov):
    #x = np.atleast_2d(x)
    x_mu = x - mu
    inv_convmat = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    dist = np.sqrt(np.sum(np.dot(x_mu, inv_convmat) * x_mu, axis=1))
    #mean = np.atleast_2d(mu)
    #diff = x - mu
    #dist = np.sqrt(np.sum(np.dot(diff, cov_inv) * diff, axis=1))
    print("Distância de Mahalanobis:", dist)
    #print(f"x: {x} \n/ mu: {mu}\n / x_mu: {x_mu}")
    #print(f"inv_convmat: ", inv_convmat)
    return dist
    
    

def adjust_threshold(features, mu, cov, percentile=95):
    distances = mahalonobis_distance(features, mu, cov)
    threshold = np.percentile(distances, percentile)
    #print(f"Threshold ajustado (percentil {percentile}): {threshold:.2f}")
    return threshold              

# Verifica a anomalia
def check_anomaly(features, mean, cov_inv, threshold):
    #features = extract_fft_features(sample)
    dist = mahalonobis_distance(features, mean, cov_inv)
    print(f"VALORES CHECK_ANOMALY: \n DIST: {dist} \n THRESHOLD: {threshold}")
    return dist, dist > threshold