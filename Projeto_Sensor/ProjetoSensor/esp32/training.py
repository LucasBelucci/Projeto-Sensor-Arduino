# Import e Setup

import os
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (classification_report, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import joblib

'''
Importações realizadas para testar a curva ROC e a distribuição de distâncias

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, accuracy_score,
                             roc_auc_score, roc_curve)
'''

## Configurações de DATASET
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

## Hiperparâmetros
VAL_RATIO = 0.2
TEST_RATIO = 0.2
MAX_ANOMALY_SAMPLES = 100
MAX_NORMAL_SAMPLES = 100

## Caminho do modelo
MODEL_PATH = Path("models/mahalonobis_model.npz")

# %%
def get_data_files(operations: list[str]) -> list[Path]:
    """
    Retorna a lista de arquivos CSV em cada diretório de operação.

    Args:
        operations (list): Lista de nomes de pastas com dados.

    Returns:
        list: Retorna uma Lista com os caminhos para os arquivos CSV.
    """
    files = []
    for op in operations:
        files.extend(list((DATASET_PATH / op).glob("*.csv")))
    return files

def mahalonobis_distance(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Calcula a distância de Mahalonobis entre os vetores 'x' e a média 'mu', com base na matriz de covariância 'cov'.

    Args:
        x (np.ndarray): Vetores de entrada, shape (n amostras, n features).
        mu (np.ndarray): Vetor de média, shape (n features).
        cov (np.ndarray): Matriz de covariância, shape (n features, n features).

    Returns:
        np.ndarray: Distância de Mahalanobis de cada amostra em 'x'.
    """
    x_mu = np.atleast_2d(x - mu)

    if np.ndim(cov) == 1:
        cov = np.array([[cov]])

    inv_convmat = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    return np.sqrt(np.sum(np.dot(x_mu, inv_convmat) * x_mu, axis=1))

# Para extração de algumas características importantes, é realizado a aplicação da janela de Hann, minimizando os efeitos de descontinuidade nas extremidades
# dos sinais, sabendo que estamos utilizando sinais reais, a FFT será simétrica, e portanto, apenas a primeira metade dos valores contém informações úteis
# Aplicando então a janela nos dados antes de realizar a FFT, calculando apenas a parte positiva, extraindo a magnitude e ignorando as fases, 
# será obtido uma matriz com as frequências extraídas de cada eixo após ignorar a simetria

def extract_fft_features(sample: np.ndarray, include_dc: bool = False, window: str = "hann", return_freqs: bool = False, sampling_rate: float = 1.0) -> np.ndarray | Tuple[np.ndarray, np.ndarray]: 
    """
    Extrai caracteristicas de FFT de um sinal multivariado

    Args:
        sample(np.ndarray): Sinal com shape (n_samples, n_axes)
        include_dc(bool): Se True, inclui a componente DC(frequência 0)
        window(str): Tipo de janela a ser aplicada('hann', 'hamming', None)
    
    Returns:
        np.ndarray: FFT do sinal com shape(n_samples//2, n_axes)
        np.ndarray: Frequências associadas a cada componente da FFT
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
    #corr_matrix = np.corrcoef(sample.T) # shape: (n_axes, n_axes)
    #tril_indices = np.tril_indices_from(corr_matrix, k=-1)
    #features.append(corr_matrix[tril_indices])  # shape: (n_combinations, )

    # Domínio da frequência
    #fft = extract_fft_features(sample, include_dc=False)    # shape: (n_freqs, n_axes)

    #features.append(np.mean(fft, axis=0))   # Média das magnitudes FFT
    #features.append(np.std(fft, axis=0))    # Desvio padrão das FFT
    #features.append(np.sum(fft**2, axis=0)) # Energia
    #features.append(np.argmax(fft, axis=0)) # Freq. Dominante (índice de máxima magnitude)

    # Concatena tudo em um vetor 1D
    return np.concatenate(features)

def load_and_extract_file(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega o arquivo CSV contendo os sinais de sensores, remove a média DC, adiciona o ruído aleatório e extrai características no domínio da frequência (FFT).

    Args:
        file_path (Path): caminho para o arquivo a ser processado.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Matriz com as componentes FFT (shape: n_samples//2, n_axes)
            - Frequências associadas a cada componente da FFT
    """
    data = np.genfromtxt(file_path, delimiter=",")
    data = data - np.mean(data, axis=0)

    # Adicionando ruído para melhorar o treinamento

    noise = np.random.normal(0, 0.3, data.shape)
    data = data + noise

    return extract_fft_features(data)

def create_dataset(files: list[Path], max_samples: int=50) -> np.ndarray:
    """
    Processa uma lista de arquivos CSV contendo sinais de sensores. Para cada arquivo, aplica a remoção da média DC, 
    adiciona ruído aleatório e extrai características no domínio da frequência (FFT) por meio da função 
    `load_and_extract_file`.

    Se o número de arquivos fornecidos exceder `max_samples`, um subconjunto aleatório será selecionado.

    Args:
        files (list[Path]): Lista de caminhos para os arquivos CSV a serem processados.
        max_samples (int, optional): Número máximo de arquivos a serem utilizados. Valor padrão é 50.

    Returns:
        np.ndarray: Array contendo os vetores de características extraídas de cada arquivo.
    """
    # Seleciona arquivos aleatórios se exceder o número máximo permitido
    if len(files) > max_samples:
        files = np.random.choice(files, max_samples, replace=False)

    features = [load_and_extract_file(f) for f in files]
    return np.array(features)

'''
def plot_roc_curve(normal_distances, anomaly_distances, save_path=None):
    y_true = np.concatenate(
        [np.zeros(len(normal_distances)), np.ones(len(anomaly_distances))]
    )
    distances = np.concatenate([normal_distances, anomaly_distances])

    fpr, tpr, _ = roc_curve(y_true, distances)
    auc = roc_auc_score(y_true, distances)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, label=f"Curva ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0,1], "k--", label="Random")
    ax.set_xlabel("Taxa de Falsos positivos")
    ax.set_ylabel("Taxa de Verdadeiros positivos")
    ax.set_title("Curva ROC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    #if save_path:
    #    salvar_figura(save_path)
    #return fig

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pd.DataFrame(cm, index=["Normal", "Anomaly"], columns=["Normal", "Anomaly"]),
                     annot=True, fmt="d", cmap="Blues",
    )
    
    ax.set_title("Matriz de Confusão")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    #if save_path:
    #    salvar_figura(save_path)
    return fig
'''
def find_optimal_threshold(normal_dist: Path, anomaly_dist: Path, n_splits: int=5) -> Tuple[np.ndarray, dict]:
    """
    Encontra o limiar (threshold) ótimo para classificar anomalias com base em distribuições de distância.

    Args:
        normal_dist (np.ndarray): Vetor de distâncias de amostras normais.
        anomaly_dist (np.ndarray): Vetor de distâncias de amostras anômalas.
        n_splits (int, optional): Número de iterações de validação (simulação de cross-validation). Padrão é 5.

    Returns:
        Tuple[np.ndarray, dict]: 
            - Threshold ótimo encontrado (float)
            - Dicionário com métricas médias (fp_rate e tp_rate)
    """
    normal_range = np.percentile(normal_dist, [75, 99])
    anomaly_range = np.percentile(anomaly_dist, [1, 25])

    thresholds = np.linspace(normal_range[0], anomaly_range[1], 100)

    best_score = -np.inf
    best_threshold = None
    best_metrics = None

    for threshold in thresholds:
        fold_metrics = []
        for _ in range(n_splits):
            normal_mask = np.random.choice(
                [True, False], len(normal_dist), p=[0.7, 0.3]
            )
            anomaly_mask = np.random.choice(
                [True, False], len(anomaly_dist), p=[0.7, 0.3]
            )

            normal_pred = normal_dist[normal_mask] > threshold
            anomaly_pred = anomaly_dist[anomaly_mask] > threshold

            fp_rate = np.mean(normal_pred)
            tp_rate = np.mean(anomaly_pred)

            # Penalização para resultados muito altos
            penalizacao = 3

            score = tp_rate - (penalizacao * fp_rate)

            # Penalização para resultados perfeitos, indicando overfitting

            if fp_rate == 0 or tp_rate == 1:
                score *= (
                    0.5
                )

            fold_metrics.append(
                {"score": score, "fp_rate": fp_rate, "tp_rate": tp_rate}
            )

        avg_score = np.mean([m["score"] for m in fold_metrics])
        score_std = np.std([m["score"] for m in fold_metrics])

        # Preferência por resultados mais estáveis

        final_score = avg_score - (2 * score_std)

        if final_score > best_score:
            best_score = final_score
            best_threshold = threshold
            best_metrics = {
                "fp_rate": np.mean([m["fp_rate"] for m in fold_metrics]),
                "tp_rate": np.mean([m["tp_rate"] for m in fold_metrics])
            }

    return best_threshold, best_metrics

def train_model() -> None:
    """
    Treina um modelo de detecção de anomalias baseado em distância de Mahalanobis, usando dados de sensores.

    O processo inclui:
    - Carregamento de dados normais e anômalos de múltiplos arquivos CSV.
    - Extração de características no domínio da frequência (FFT) e estatísticas.
    - Escalonamento dos dados com StandardScaler.
    - Cálculo da média e matriz de covariância dos dados normais de treino.
    - Definição de um limiar com 5% de taxa de falso positivo baseado na distribuição de distâncias.
    - Avaliação do modelo com métricas de classificação e AUC.
    - Salvamento dos artefatos treinados: modelo (mu, cov, threshold), scaler e resultados intermediários.

    Retorna:
        None
    """
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
    
    # Cálculo da média e da covariância
    mu = np.mean(X_train_scaled, axis=0)
    cov = np.cov(X_train_scaled.T)

    if np.ndim(cov) == 0:
        cov = np.array([[cov]])

    # Encontrando o limiar com 5% de taxa de falso positivo
    normal_dist = mahalonobis_distance(X_test_scaled, mu, cov)
    anomaly_dist = mahalonobis_distance(X_anomaly_scaled, mu, cov)
    threshold, metrics = find_optimal_threshold(normal_dist, anomaly_dist)
    #threshold = np.percentile(normal_dist, 95)
    print(f"\nThreshold encontrado: {threshold:.3f}")
    print(f"Taxa de falso positivo (FP rate): {metrics['fp_rate']:.2%}")
    print(f"Taxa de verdadeiro positivo (TP rate): {metrics['tp_rate']:.2%}")
    


    # Avaliação
    y_true = np.concatenate([np.zeros(len(X_test)), np.ones(len(X_anomaly))])
    y_pred = np.concatenate([normal_dist > threshold, anomaly_dist > threshold]).astype(int)

    #plot_roc_curve(normal_dist, anomaly_dist, None)
    #plt.show()

    #plot_confusion_matrix(y_true, y_pred, None)
    #plt.show()

    # Mostrando os resultados através dos plots e dos prints
    #print("\nValores usados: ")
    #print(f"\n{X_train_scaled}")
    #print(f"\n{X_test_scaled}")
    #print(f"\n{X_anomaly_scaled}")
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