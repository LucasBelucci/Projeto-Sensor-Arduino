# Import e Setup

from pathlib import Path
import os
import numpy as np
from scipy import stats
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score)
from joblib import dump
import joblib

DATASET_PATH = Path("ProjetoSensor/datasets/ac")
NORMAL_OPS = ["silent_0_baseline"]
ANOMALY_OPS = [
    "medium_0",
    "high_0",
    "silent_1",
    "medium_1",
    "high_1"
]
SAMPLE_RATE = 200
SAMPLE_TIME = 0.5

# Valores para treinamento

VAL_RATIO = 0.2
TEST_RATIO = 0.2
MAX_ANOMALY_SAMPLES = 100
MAX_NORMAL_SAMPLES = 100
MODEL_PATH = Path("models/mahalonobis_model.npz")

# %%
def get_data_files(operations):
    files = []
    for op in operations:
        files.extend(list((DATASET_PATH / op).glob("*.csv")))
    return files

def mahalonobis_distance(x, mu, cov):
    x_mu = np.atleast_2d(x - mu)

    if np.ndim(cov) == 1:
        cov = np.array([[cov]])

    inv_convmat = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    return np.sqrt(np.sum(np.dot(x_mu, inv_convmat) * x_mu, axis=1))

# Para extração de algumas características importantes, é realizado a aplicação da janela de Hann, minimizando os efeitos de descontinuidade nas extremidades
# dos sinais, sabendo que estamos utilizando sinais reais, a FFT será simétrica, e portanto, apenas a primeira metade dos valores contém informações úteis
# Aplicando então a janela nos dados antes de realizar a FFT, calculando apenas a parte positiva, extraindo a magnitude e ignorando as fases, 
# será obtido uma matriz com as frequências extraídas de cada eixo após ignorar a simetria

def extract_fft_features(
        sample: np.ndarray,
        include_dc: bool = False,
        window: str = "hann",
        return_freqs: bool = False,
        sampling_rate: float = 1.0
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    
    """
    Extrai caracteristicas de FFT de um sinal multivariado

    Args:
        sample(np.ndarray): Sinal com shape (n_samples, n_axes)
        include_dc(bool): Se True, inclui a componente DC(frequência 0)
        window(str): Tipo de janela a ser aplicada('hann', 'hamming', None)
    
    Returns:
        np.ndarray: FFT do sinal com shape(n_samples//2, n_axes)
        np.ndarray(opcional): Frequências associadas a cada componente da FFT
    """

    if not isinstance(sample, np.ndarray):
            raise TypeError("Entrada deve ser um np.ndarray")
        
    if sample.ndim != 2:
        raise ValueError("Entrada deve ter 2 dimensões")
    
    n_samples, n_axes = sample.shape # sample.shape[0] = dimensão // sample.shape[1] = amostras
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
    
    if n_samples < 2:
        raise ValueError("Número de amostras deve ser maior ou igual a 2")
    
    # Seleção da janela:
    if window == 'hann':
        window_values = np.hanning(n_samples)
    elif window == 'hamming':
        window_values = np.hamming(n_samples)
    elif window is None:
        window_values = np.ones(n_samples)
    else:
        raise ValueError(f'Tipo de janela nao suportado: {window}')
    
    # Tamanho da FFT (Considerando ou não a componente DC)
    fft_len = len(np.fft.rfft(sample[:, 0] * window_values))
    if not include_dc:
        fft_len -= 1

    # Aplicando uma janela FFT em cada eixo da amostra e ignorando a componente DC
    out_sample = np.zeros((fft_len, n_axes))
    output_freqs = freqs if include_dc else freqs[1:]

    for i, axis in enumerate(sample.T):
        fft = np.abs(np.fft.rfft(axis * window_values)) / n_samples # Normalizando a FFT, garantindo a compatibilidade entre diferentes sinais ou janelas
        if n_samples % 2 == 0:
            fft[1:-1] *= 2 # par: dobra todas, exceto DC e Nyquist
        else:
            fft[1:] *= 2    # impar: dobra todas, exceto DC (Não tem Nyquist exato)
        out_sample[:, i] = fft if include_dc else fft[1:] # Ignorando a componente DC
    
    if return_freqs:
        return out_sample, output_freqs
    else:
        return out_sample

# Extração das estatísticas para utilização em ML
def extract_ml_features(sample: np.ndarray) -> np.ndarray:
    """
    Extrai features estatísticas e de frequência de um sinal multivariado

    Args:
        sample(np.ndarray): Sinal de entrada com shape(n_samples, n_axes)

    Returns:
        np.ndarray: Vetor 1D com as features extraidas
    """

    if not isinstance(sample, np.ndarray) or sample.ndim != 2:
        raise ValueError("Entrada precisa ser um np.ndarray 2D")
    
    features = []

    # Domínio do tempo
    features.append(np.mean(sample, axis=0)) # shape: (n_axes, )
    features.append(np.var(sample, axis=0))
    features.append(stats.skew(sample, axis=0))
    features.append(stats.kurtosis(sample, axis=0))
    features.append(np.mean(np.abs(sample - np.mean(sample, axis=0)), axis=0))

    # Correlação entre eixos
    corr_matrix = np.corrcoef(sample.T) # shape: (n_axes, n_axes)
    tril_indices = np.tril_indices_from(corr_matrix, k=-1)
    features.append(corr_matrix[tril_indices])  # shape: (n_combinations, )

    # Domínio da frequência
    fft = extract_fft_features(sample, include_dc=False)    # shape: (n_freqs, n_axes)

    features.append(np.mean(fft, axis=0))   # Média das magnitudes FFT
    features.append(np.std(fft, axis=0))    # Desvio padrão das FFT
    features.append(np.sum(fft**2, axis=0)) # Energia
    features.append(np.argmax(fft, axis=0)) # Freq. Dominante (índice de máxima magnitude)

    # Concatena tudo em um vetor 1D
    return np.concatenate(features)

def load_and_extract_files(file_path):
    data = np.genfromtxt(file_path, delimiter=",")
    data = data - np.mean(data, axis=0)

    # Adicionando ruído para melhorar o treinamento

    noise = np.random.normal(0, 0.3, data.shape)
    data = data + noise

    return extract_fft_features(data)

def create_dataset(files, max_samples=50):

    # Se tiver mais arquivos do que o max_samples, eles serão escolhidos de maneira aleatória

    if len(files) > max_samples:
        files = np.random.choice(files, max_samples, replace=False)

    features = [load_and_extract_files(f) for f in files]
    return np.array(features)

def train_model():
    # Carrega e prepara os dados 

    normal_files = get_data_files(NORMAL_OPS)
    anomaly_files = get_data_files(ANOMALY_OPS)
    print(f"Encontrado: {len(normal_files)} arquivos normais e {len(anomaly_files)} arquivo anomalos")

    # Separa os dados normais

    train_files, test_files = train_test_split(normal_files, test_size=0.4, random_state=42)

    # Criação dos datasets

    X_train = create_dataset(train_files)
    X_test = create_dataset(test_files)
    X_anomaly = create_dataset(anomaly_files)

    X_train_raw = np.array([extract_ml_features(np.atleast_2d(x)) for x in X_train])
    X_test_raw = np.array([extract_ml_features(np.atleast_2d(x)) for x in X_test])
    X_anomaly_raw = np.array([extract_ml_features(np.atleast_2d(x)) for x in X_anomaly])

    # Features do Scaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    X_anomaly_scaled = scaler.transform(X_anomaly_raw)
    
    # Modelo de treino

    mu = np.mean(X_train_scaled, axis=0)
    cov = np.cov(X_train_scaled.T)

    if np.ndim(cov) == 0:
        cov = np.array([[cov]])

    # Encontrando o limiar com 5% de taxa de falso positivo

    normal_dist = mahalonobis_distance(X_test_scaled, mu, cov)
    anomaly_dist = mahalonobis_distance(X_anomaly_scaled, mu, cov)
    threshold = np.percentile(normal_dist, 95)


    # Avaliação

    y_true = np.concatenate([np.zeros(len(X_test)), np.ones(len(X_anomaly))])
    y_pred = np.concatenate([normal_dist > threshold, anomaly_dist > threshold]).astype(int)

    # Mostrando os resultados através dos plots e dos prints
    print("\nValores usados: ")
    print(f"\n{X_train_scaled}")
    print(f"\n{X_test_scaled}")
    print(f"\n{X_anomaly_scaled}")
    #print(f"normal_dist:{normal_dist}\nanomaly_dist:{anomaly_dist}\n")
    print("\nResultado da Classificação:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
    print(f"AUC Score: {roc_auc_score(y_true, np.concatenate([normal_dist, anomaly_dist])):.3f}")

    # Salvando o modelo de treinamento gerado

    os.makedirs("models", exist_ok=True)
    np.savez(MODEL_PATH, mu=mu, cov=cov, threshold=threshold)
    print(f"\nModelo salvo em {MODEL_PATH}")

    # Salvando os objetos
    dump(normal_dist, "models/normal_dist.joblib")
    dump(anomaly_dist, "models/anomaly_dist.joblib")
    dump(threshold, "models/threshold.joblib")
    joblib.dump(scaler, 'models/scaler.pkl')
    dump(y_true, "models/y_true.joblib")
    dump(y_pred, "models/y_pred.joblib")
    

if __name__ == "__main__":
    train_model()