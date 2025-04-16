# Import e Setup

from pathlib import Path
import os
import numpy as np
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, accuracy_score,
                             roc_auc_score, roc_curve)
from analysis import (plot_3d_scatter, plot_comparison, plot_confusion_matrix, plot_distance_distributions,
                      plot_fft_comparison, plot_histograms, plot_roc_curve, extract_fft_features, extract_ml_features,
                      mahalonobis_distance)

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

def load_and_extract_files(file_path):
    data = np.genfromtxt(file_path, delimiter=",")
    data = data - np.mean(data, axis=0)

    # Adicionando ruído para melhorar o treinamento

    noise = np.random.normal(0, 0.3, data.shape)
    data = data + noise

    # Extraindo features por eixo
    '''
    features = []
    for axis_idx in range(data.shape[1]):
        axis_data = data[:, axis_idx]
        features.extend(
            [
                np.std(axis_data),
                scipy_stats.kurtosis(axis_data),
                np.max(np.abs(axis_data)),
                np.sqrt(np.mean(np.square(axis_data))),
                np.max(axis_data) - np.min(axis_data)
            ]
        )
    return np.array(features)
    '''
    return extract_ml_features(data)

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

    # Features do Scaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_anomaly_scaled = scaler.transform(X_anomaly)

    # Modelo de treino

    mu = np.mean(X_train_scaled, axis=0)
    cov = np.cov(X_train_scaled.T)

    # Encontrando o limiar com 5% de taxa de falso positivo

    normal_dist = mahalonobis_distance(X_test_scaled, mu, cov)
    anomaly_dist = mahalonobis_distance(X_anomaly_scaled, mu, cov)
    threshold = np.percentile(normal_dist, 95)

    # Avaliação

    y_true = np.concatenate([np.zeros(len(X_test)), np.ones(len(X_anomaly))])
    y_pred = np.concatenate([normal_dist > threshold, anomaly_dist > threshold]).astype(int)

    # Mostrando os resultados através dos plots e dos prints

    print("\nResultado da Classificação:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
    print(f"AUC Score: {roc_auc_score(y_true, np.concatenate([normal_dist, anomaly_dist])):.3f}")

    plot_distance_distributions(normal_dist, anomaly_dist, threshold)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(normal_dist, anomaly_dist)

    plt.show()
    
    # Salvando o modelo de treinamento gerado

    os.makedirs("models", exist_ok=True)
    np.savez(MODEL_PATH, mu=mu, cov=cov, threshold=threshold, scaler=scaler)
    print(f"\nModelo salvo em {MODEL_PATH}")

if __name__ == "__main__":
    train_model()