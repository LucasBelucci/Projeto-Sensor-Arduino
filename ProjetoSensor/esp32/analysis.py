# %% Setup
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
import seaborn as sns
from scipy import stats
from matplotlib.figure import Figure
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, accuracy_score,
                             roc_auc_score, roc_curve)

from training import extract_fft_features
from joblib import load
from datetime import datetime


# %% Definição dos caminhos, nomes das pastas, taxa de amostragem em Hz e em segundos
DATASET_PATH = Path("ProjetoSensor/datasets/ac")
NORMAL_OPS = ["silent_0_baseline"]
ANOMALY_OPS = ["medium_0", "high_0", "silent_1", "medium_1", "high_1"]
SAMPLE_RATE = 200
SAMPLE_TIME = 0.5

# %% Códigos responsáveis por identificar todos os arquivos que serão utilizados e realizar o carregamento dos dados que serão utilizados com uma remoção opcional do DC
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
        path = DATASET_PATH / op
        files.extend(list(path.glob("*.csv")))
    return files

def load_samples(file_path: Path, remove_dc: bool = False) -> np.ndarray | None:
    """
    Carrega o arquivo CSV contendo os sinais de sensores e, opcionalmente, remove a média DC de cada canal.

    Args:
        file_path (Path): caminho para o arquivo a ser processado.
        remove_dc (bool, optional): Se True, remove a média DC de cada componente, por padrão é definido como False.

    Returns:
        np.ndarray | None: Matriz com os dados do sensor ou None em caso de erro.
    """
    try:
        data = np.genfromtxt(file_path, delimiter=",")
        if remove_dc:
            data = data - np.mean(data, axis=0)
        return data
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        return None
    
def gerar_nome_arquivo(tipo: str, sufixo: str = "", extensao: str = "png", timestamp: str = None) -> Path:
    """
    Gera um nome de arquivo com base no tipo, sufixo, extensão e timestamp atual (ou fornecido).

    Args:
        tipo (str): Prefixo descritivo do tipo de gráfico ou imagem.
        sufixo (str): Informação adicional para o nome do arquivo, por padrão é "".
        extensão (str): Extensão do arquivo, por padrão é "png".
        timestamp (str, optional): Timestamp formatado (ex: "20250603_202328"). Se None, utiliza o timestamp atual.

    Returns:
        Path: Caminho para o arquivo dentro da pasta "Imagens/graficos".
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome = f"{tipo}_{sufixo}_{timestamp}" if sufixo else f"{tipo}_{timestamp}"
    return Path("Imagens/graficos") / f"{nome}.{extensao}"

def salvar_figura(save_path: Path) -> None:
    """
    Salva a figura matplotlib atual no caminho especificado.

    Além disso, garante que o diretório de destino exista e caso ocorra um erro no salvamento, uma mensagem de erro será exibida.

    Args:
        save_path (Path): Caminho completo do arquivo onde a figura será salva, incluindo o nome e extensão.

    Returns:
        None
    """
    save_path = Path(save_path)
    try:
        # Garante que a pasta existe
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Tenta salvar a figura
        plt.savefig(str(save_path), dpi=300, bbox_inches = "tight")

        # Verifica se o arquivo foi salvo
        if Path(save_path).exists():
            print(f"[SUCESSO] Imagem salva em: {save_path.resolve()}")
        else:
            print(f"[ERRO] plt.savefig() executou mas o arquivo nao apareceu!")     
    except Exception as e:
        print(f"[ERRO] Não foi possível salvar a imagem: {e}")
    finally:
        plt.close()

# Funções responsáveis pelos gráficos comparativos que serão gerados entre as informações normais e as anomalias, através de gráficos de linha, dispersão em 3D
def plot_comparison(normal_file: Path, anomaly_file: Path, remove_dc: bool=False, save_path: Path=None) -> Figure:
    """
    Realiza o plot e compara os sinais de operação normal e anormal de sensores em gráficos separados.

    Os sinais podem ter o componente DC removido e o gráfico gerado pode ser salvo no caminho fornecido.

    Args:
        normal_file (Path): Caminho para o arquivo CSV contendo dados normais.
        anomaly_file (Path): Caminho para o arquivo CSV contendo dados anômalos.
        remove_dc (bool, optional): Se True, remove a média DC dos sinais antes da plotagem.
        save_path (Path, optional): Caminho para salva a figura gferada, se None apenas exibe o gráfico.

    Returns:
        matplotlib.figure.Figure: Objeto da figura gerada.
    """
    normal_data = load_samples(normal_file, remove_dc)
    anomaly_data = load_samples(anomaly_file, remove_dc)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        "Comparação dos dados do acelerômetro" + (" (DC removida)" if remove_dc else ""),
        fontsize=14,
        y=1.02,
    )

    for i, axis in enumerate(["X", "Y", "Z"]):
        ax1.plot(normal_data[:, i], label=f"{axis}-axis", linewidth=2)
    ax1.set_title("Operação Normal", pad=10)
    ax1.set_ylabel("Força G")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for i, axis in enumerate(["X", "Y", "Z"]):
        ax2.plot(anomaly_data[:, i], label=f"{axis}-axis", linewidth=2)
    ax2.set_title("Operação Anormal", pad=10)
    ax2.set_ylabel("Força G")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        salvar_figura(save_path)
    return fig

def plot_3d_scatter(normal_files: list[Path], anomaly_files: list[Path], num_samples: int=3, feature_type: str="raw", save_path: Path=None) -> Figure:
    """
    Plota um gráfico 3D comparando dados normais e anômalos do acelerômetro, com base em diferentes tipos de features.

    Os dados podem ser processados como brutos ("raw") ou transformados via estatísticas como média, variância, etc.
    Apenas os primeiros 'num_samples' arquivos de cada tipo são considerados.

    Args:
        normal_files (list[Path]): Lista de caminhos para arquivos CSV com dados normais.
        anomaly_files (list[Path]): Lista de caminhos para arquivos CSV com dados anômalos.
        num_samples (int, optional): Número de amostras a serem utilizadas (padrão = 3).
        feature_type (str, optional): Tipo de feature a ser extraída. Pode ser:
            - "raw": usa os dados brutos.
            - "mean", "variance", "kurtosis", "entropy", "energy": aplica a função correspondente.
        save_path (Path, optional): Caminho para salvar a figura gerada. Se None, a figura não será salva.

    Returns:
        matplotlib.figure.Figure: Objeto da figura gerada.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    normal_data = []
    anomaly_data = []

    for i in range(min(num_samples, len(normal_files))):
        normal_sample =load_samples(normal_files[i], remove_dc=True)
        anomaly_sample =load_samples(anomaly_files[i], remove_dc=True)

        if feature_type == "mean":
            normal_data.append(np.mean(normal_sample, axis=0))
            anomaly_data.append(np.mean(anomaly_sample, axis=0))
        elif feature_type == "variance":
            normal_data.append(np.var(normal_sample, axis=0))
            anomaly_data.append(np.var(anomaly_sample, axis=0))
        elif feature_type == "kurtosis":
            normal_data.append(stats.kurtosis(normal_sample))
            anomaly_data.append(stats.kurtosis(anomaly_sample))
        elif feature_type == "entropy":
            normal_data.append(stats.entropy(np.abs(normal_sample)))
            anomaly_data.append(stats.entropy(np.abs(anomaly_sample)))
        elif feature_type == "energy":
            normal_data.append(np.sum(np.square(normal_sample), axis=0))
            anomaly_data.append(np.sum(np.square(anomaly_sample), axis=0))
        else:
            normal_data.append(normal_sample)
            anomaly_data.append(anomaly_sample)

    if feature_type in ['mean', 'variance', 'kurtosis', "entropy", "energy"]:
        normal_data = np.array(normal_data)
        anomaly_data = np.array(anomaly_data)

        ax.scatter(
            normal_data[:, 0],
            normal_data[:, 1],
            normal_data[:, 2],
            alpha=0.6,
            label="Normal",
        )

        ax.scatter(
            anomaly_data[:, 0],
            anomaly_data[:, 1],
            anomaly_data[:, 2],
            alpha=0.6,
            label="Anomaly",
        )

    else:
        for i in range(len(normal_data)):
            ax.scatter(
                normal_data[i][:, 0],
                normal_data[i][:, 1],
                normal_data[i][:, 2],
                alpha=0.6,
                label="Normal" if i == 0 else None,
            )

            ax.scatter(
                anomaly_data[i][:, 0],
                anomaly_data[i][:, 1],
                anomaly_data[i][:, 2],
                alpha=0.6,
                label="Anomaly" if i == 0 else None,
            )
    
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(f"3D Visualização de {feature_type.capitalize()} Data")
    ax.legend()

    if save_path:
        salvar_figura(save_path)

    return fig

def plot_fft(df:pd.DataFrame, st, sampling_rate: int=100, eixo: str='x', width: float=3.5, height:float=2.5) -> None:
    """
    Plota o espectro de frequência (FFT) de um eixo específico.

    Args:
        df (pd.DataFrame): DataFrame com colunas ['x', 'y', 'z'].
        st (streamlit module): Módulo streamlit.
        eixo (str): Eixo a ser plotado ('x', 'y' ou 'z').
        width (float): Largura da figura.
        height (float): Altura da figura.

    Returns:
        matplotlib.figure.Figure: Objeto da figura gerada.
    """
    if eixo not in df.columns:
        st.warning(f"Eixo '{eixo}' não encontrado")
        return
    
    data = df[eixo].values
    n = len(data)
    window = np.hanning(n)
    data_windowed = data * window
    
    fft = np.fft.rfft(data_windowed)
    fft_magnitude = np.abs(fft) / n
    fft_magnitude[1:] *= 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(freqs, fft_magnitude, label = f'FFT - eixo {eixo.upper()}')
    ax.set_title(f'FFT - eixo {eixo.upper()}')
    ax.set_xlabel('Frequência (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def plot_feature_histogram(features: pd.DataFrame, st) -> None:
    """
    Plota um histograma de cada uma das features.

    Args:
        features (pd.Dataframe): Dataframe contendo as features a serem analisadas.
        st (módulo): Módulo do Streamlite    
    
    Returns:
        None
    """
    for col in features.columns:
        fig, ax = plt.subplots()
        ax.hist(features[col], bins=20, color='skyblue', edgecolor="black")
        ax.set_title(f"Histograma - {col}")
        st.pyplot(fig)

        plt.close(fig)

def plot_histograms(normal_files: list[Path], anomaly_files: list[Path], save_path: Path=None) -> None:
    """
    Faz o plot de histogramas utilizando as features, porém comparando para os valores nas amostras normais com as amostras anomalas.
    O gráfico pode ser salvo no caminho indicado, caso desejado.

    Args:
        normal_files (list[Path]): Lista de caminhos para arquivos CSV com dados normais.
        anomaly_files (list[Path]): Lista de caminhos para arquivos CSV com dados anômalos.
        save_path (Path, optional): Caminho para salvar a figura gerada. Se None, a figura não será salva.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))

    normal_valid = []
    anomaly_valid = []
    for f in normal_files:
        sample = load_samples(f)
        if sample is not None:
            normal_valid.append(sample)
    
    
    for f in anomaly_files:
        sample = load_samples(f)
        if sample is not None:
            anomaly_valid.append(sample)

    #print(f'Amostras normais válidas: {len(normal_valid)}')
    #print(f'Amostras anormais válidas: {len(anomaly_valid)}')

    if not normal_valid or not anomaly_valid:
        raise ValueError("Não há amostras válidas para plotar")
    
    num_features = normal_valid[0].shape[1]
    axis_labels = ["X-axis", "Y-axis", "Z-axis"]
    
    for i in range(num_features):
        plt.subplot(2, 3, i+1)
        
        normal_feature_values = [sample[:, i] for sample in normal_valid]
        anomaly_feature_values = [sample[:, i] for sample in anomaly_valid]

        normal_flat = np.concatenate(normal_feature_values)
        anomaly_flat = np.concatenate(anomaly_feature_values)

        sns.histplot(normal_flat, color='blue', label='Normal', kde=True)
        sns.histplot(anomaly_flat, color="red", label="Anomaly", kde=True)
        plt.legend()
        plt.title(axis_labels[i])
    
    plt.tight_layout()

    if save_path is not None:
        salvar_figura(save_path)
    else:
        plt.show()

# Definição das estatísticas básicas que serão utilizadas como critérios de avaliação de desempenho dos algoritmos de aprendizado, são elas Média, variância, Kurtosis, Skew, MAD e correlação

def analyze_statistics(sample_file:Path) -> dict[str, Any]:
    """
    Realiza o carregamento dos dados do arquivo CSV, determina as estatísticas alvo e salva os resultados em um dicionário.

    Args:
        sample_file (Path): Arquivo CSV com os dados a serem utilizados.

    Returns:
        dict (str,): Dicionário de strings com os resultados obtidos nas estatísticas alvo
    """
    sample = load_samples(sample_file, remove_dc=True)

    if sample is None:
        raise ValueError(f"Não foi possível carregar amostra de {sample_file}")

    stats_dict = {
        #"Sample shape": sample.shape, # Retorna as dimensões do conjunto de dados
        "Mean": np.mean(sample, axis=0), # retorna a média de cada eixo, para o caso, o valor médio da aceleração nos eixos
        "Variance": np.var(sample, axis=0), # Mede a dispersão dos dados em torno da média
        "Kurtosis": stats.kurtosis(sample), # Indica o achatamento da distribuição de dados
        "Skew": stats.skew(sample), # Skew, também conhecido como assimetria, indica a simetria da distribuição dos dados
        "MAD": stats.median_abs_deviation(sample), # Determina a mediana das diferenças absolutas entre os valores e a mediana dos dados
        "Correlation": np.corrcoef(sample.T), # Obtem a relação linear entre os diferentes eixos (X, Y, Z) do acelerômetro
    }

    return stats_dict
        
# Geração de um gráfico comparativo das frequências entre operações normais e anômalas, ideal para identificar padrões de vibração atípicos

def plot_fft_comparison(normal_files: list[Path], anomaly_files: list[Path], num_samples: int=200, start_bin: int=1, save_path: Path=None) -> Figure:
    """
    Compara os espectros de frequência médios entre amostras normais e anômalas para os 3 eixos.

    Args:
        normal_files (list[Path]): Caminhos para arquivos CSV com dados normais.
        anomaly_files (list[Path]): Caminhos para arquivos CSV com dados anômalos.
        num_samples (int): Quantidade máxima de amostras a considerar de cada tipo.
        start_bin (int): Índice inicial da FFT a ser exibido (ignora a componente DC se > 0).
        save_path (Path, optional): Caminho para salvar a figura. Se None, a figura não é salva.

    Returns:
        matplotlib.figure.Figure: Figura matplotlib contendo os gráficos comparativos.
    """
    normal_ffts = []
    anomaly_ffts = []

    if len(normal_files) < num_samples or len(anomaly_files) < num_samples:
        raise ValueError("Número de arquivos insuficientes para o valor de num_samples")
    
    for i in range(min(num_samples, len(normal_files))):
        normal_sample = load_samples(normal_files[i])
        anomaly_sample = load_samples(anomaly_files[i])
        normal_ffts.append(extract_fft_features(normal_sample))
        anomaly_ffts.append(extract_fft_features(anomaly_sample))

    
    normal_ffts = np.array(normal_ffts)
    anomaly_ffts = np.array(anomaly_ffts)
    normal_fft_avg = np.average(normal_ffts, axis=0)
    anomaly_fft_avg = np.average(anomaly_ffts, axis=0)

    fig, axs = plt.subplots(3, 1, figsize=(10,12))
    fig.suptitle("FFT Analysis by Axis", fontsize=14, y=1.02)

    titles = ["X-axis", "Y-axis", "Z-axis"]

    for i, ax in enumerate(axs):
        ax.plot(normal_fft_avg[start_bin:, i], label="Normal", color="blue")
        ax.plot(anomaly_fft_avg[start_bin:, i], label="Anomaly", color="red")
        ax.set_title(titles[i])
        ax.set_xlabel("Frequency Bins")
        ax.set_ylabel("Magnitude")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        salvar_figura(save_path)
    return fig

# --------------- ##
    
def validate_model(normal_distances: np.ndarray, anomaly_distances: np.ndarray, threshold: float) -> dict[str, float]:
    """
    Avalia o desempenho de um modelo de detecção de anomalias com base nas distâncias e um limiar.

    Compara as distâncias de exemplos normais e anômalos em relação a um threshold fornecido,
    gerando métricas de desempenho como acurácia, precisão, recall, F1-score, AUC e taxa de falsos positivos.

    Args:
        normal_distances (np.ndarray): Array com as distâncias Mahalanobis dos exemplos normais.
        anomaly_distances (np.ndarray): Array com as distâncias Mahalanobis dos exemplos anômalos.
        threshold (float): Valor de corte para classificar uma amostra como anômala.

    Returns:
        dict[str, float]: Dicionário contendo as métricas de avaliação:
            - 'accuracy': Acurácia do modelo.
            - 'precision': Precisão para classe anômala.
            - 'recall': Sensibilidade (taxa de verdadeiros positivos).
            - 'f1': F1-score.
            - 'auc': Área sob a curva ROC.
            - 'false_positive_rate': Taxa de falsos positivos entre os exemplos normais.
    """
    y_true = np.concatenate(
        [np.zeros(len(normal_distances)), np.ones(len(anomaly_distances))]
    )
        
    distances = np.concatenate([normal_distances, anomaly_distances])
    y_pred = (distances > threshold).astype(int)

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, distances)
    }

    # Calculo da taxa de falsos positivos

    fp = np.sum((y_true == 0) & (y_pred == 1))
    results["false_positive_rate"] = fp/len(normal_distances)

    return results
    
def plot_distance_distributions(normal_dist: np.ndarray, anomaly_dist: np.ndarray, threshold: float=None, save_path: Path=None) -> Figure:
    """
    Plota a distribuição das distâncias de Mahalanobis para amostras normais e anômalas, com opção de mostrar o limiar (threshold) e salvar a figura.

    Args:
        normal_dist (np.ndarray): Distâncias Mahalanobis das amostras normais.
        anomaly_dist (np.ndarray): Distâncias Mahalanobis das amostras anômalas.
        threshold (float, optional): Limiar para classificar uma amostra como anômala. Se None, o limiar não será exibido.
        save_path (Path, optional): Caminho para salvar a figura. Se None, a figura não será salva.

    Returns:
        matplotlib.figure.Figure: Figura matplotlib contendo os gráficos comparativos.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    n_bins = int(np.sqrt(len(normal_dist) + len(anomaly_dist)))

    ax.hist(
        normal_dist, bins=n_bins, alpha=0.7, label="Normal", color="blue", density=True
    )
    ax.hist(
        anomaly_dist, bins=n_bins, alpha=0.7, label="Anomaly", color="red", density=True
    )

    if threshold is not None:
        ax.axvline(
            x=threshold, color="k", linestyle="--", label=f"Limiar: {threshold:.2f}"
        )

    ax.set_xlabel("Distância de Mahalanobis")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição da Distância de Mahalanobis")
    ax.legend()
    ax.grid(True, alpha=0.9)

    if save_path:
        salvar_figura(save_path)

    return fig

def plot_roc_curve(normal_distances: np.ndarray, anomaly_distances: np.ndarray, save_path: Path=None) -> Figure:
    """
    Plota a curva ROC com base nas distâncias de Mahalanobis para amostras normais e anômalas, comparando o desempenho do modelo com uma classificação aleatória.

    Args:
        normal_distances (np.ndarray): Distâncias Mahalanobis das amostras normais.
        anomaly_distances (np.ndarray): Distâncias Mahalanobis das amostras anômalas.
        save_path (Path, optional): Caminho para salvar a figura. Se None, a figura não será salva.

    Returns:
        matplotlib.figure.Figure: Figura contendo a curva ROC gerada.
    """
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

    if save_path:
        salvar_figura(save_path)
    return fig

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path=None) -> Figure:
    """
    Plota a matriz de confusão com os valores de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos com base nos rótulos reais e previstos.

    Args:
        y_true (np.ndarray): Rótulos reais (0 para normal, 1 para anômalo).
        y_pred (np.ndarray): Rótulos previstos pelo modelo.
        save_path (Path, optional): Caminho para salvar a figura. Se None, a figura não será salva.

    Returns:
        matplotlib.figure.Figure: Figura contendo a curva ROC gerada.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pd.DataFrame(cm, index=["Normal", "Anomaly"], columns=["Normal", "Anomaly"]),
                     annot=True, fmt="d", cmap="Blues",
    )
    
    ax.set_title("Matriz de Confusão")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    if save_path:
        salvar_figura(save_path)
    return fig

def load_distributions() -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Carrega os arquivos serializados com as distribuições de distâncias, threshold e rótulos salvos.

    Returns:
        tuple: Contém:
            - normal_dist (np.ndarray): Distâncias Mahalanobis das amostras normais.
            - anomaly_dist (np.ndarray): Distâncias Mahalanobis das amostras anômalas.
            - threshold (float): Limiar utilizado para classificar amostras como anômalas.
            - y_true (np.ndarray): Rótulos reais (0 para normal, 1 para anômalo).
            - y_pred (np.ndarray): Rótulos previstos (com base no threshold).
    """
    # Carregando os objetos
    normal_dist = load("models/normal_dist.joblib")
    anomaly_dist = load("models/anomaly_dist.joblib")
    threshold = load("models/threshold.joblib")
    y_true = load("models/y_true.joblib")
    y_pred = load("models/y_pred.joblib")

    return normal_dist, anomaly_dist, threshold, y_true, y_pred

# %%
def main():
    # Caminho dos arquivos
    normal_files = get_data_files(NORMAL_OPS)
    anomaly_files = get_data_files(ANOMALY_OPS)

    normal_dist, anomaly_dist, threshold, y_true, y_pred = load_distributions()
    
    # Indicando o número de arquivos encontrados
    print(f"Amostras normais encontradas: {len(normal_files)}")
    print(f"Amostras anormais encontradas: {len(anomaly_files)}")

    Path("Imagens").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    # Gerando e salvando as comparações de plot com DC e sem DC
    for remove_dc in [False, True]:
        sufixo = "sem_dc" if remove_dc else "com_dc"
        output_path = gerar_nome_arquivo("comparison", sufixo = sufixo, timestamp = timestamp)
        plot_comparison(normal_files[0], anomaly_files[0], remove_dc= remove_dc, save_path=output_path)

    # Gerando e salvando outros gráficos para análises (3D e histogramas)
    output_path = gerar_nome_arquivo('raw_10', timestamp=timestamp)
    plot_3d_scatter(normal_files, anomaly_files, num_samples=10, feature_type='raw', save_path=output_path)

    feature_type = ['raw', 'mean', 'variance', 'kurtosis', 'entropy', 'energy']
    for feature in feature_type:
        output_path = gerar_nome_arquivo(feature, timestamp=timestamp)
        plot_3d_scatter(normal_files, anomaly_files, num_samples=200, feature_type=feature, save_path=output_path)
    
    # Gerando e salvando histogramas
    output_path = gerar_nome_arquivo("histogram", timestamp=timestamp)
    plot_histograms(normal_files, anomaly_files, save_path=output_path)

    # Gerando e salvando os gráficos de FFT (comparação)
    output_path= gerar_nome_arquivo("fft_comparison", timestamp=timestamp)
    plot_fft_comparison(normal_files, anomaly_files, save_path=output_path)

    # Gerando e salvando os gráficos de Distribuição das distancias
    output_path = gerar_nome_arquivo("distance_distributions", timestamp)
    plot_distance_distributions(normal_dist, anomaly_dist, threshold, save_path=output_path)
    
    # Gerando e salvando os gráficos da matriz de confusão
    output_path = gerar_nome_arquivo("confusion_matrix", timestamp)
    plot_confusion_matrix(y_true, y_pred, save_path=output_path)

    # Gerando e salvando os gráficos de ROC Curve
    output_path = gerar_nome_arquivo("roc_curve", timestamp)
    plot_roc_curve(normal_dist, anomaly_dist, save_path=output_path)
    
    plt.show()

        # Analisando estatísticas e imprimindo os resultados
    stat_results = analyze_statistics(normal_files[0])
    for key, value in stat_results.items():
        print(f"{key}:")
        print(value)
        print()

## ------------ ##

if __name__ == "__main__":
    main()
