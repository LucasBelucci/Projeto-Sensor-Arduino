# %% Setup
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import pandas as pd
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, accuracy_score,
                             roc_auc_score, roc_curve)
from ProjetoSensor.esp32.training import extract_fft_features
from joblib import load
from datetime import datetime


# %% Definição dos caminhos, nomes das pastas, taxa de amostragem em Hz e em segundos
DATASET_PATH = Path("ProjetoSensor/datasets/ac")
NORMAL_OPS = ["silent_0_baseline"]
ANOMALY_OPS = ["medium_0", "high_0", "silent_1", "medium_1", "high_1"]
SAMPLE_RATE = 200
SAMPLE_TIME = 0.5

# %% Códigos responsáveis por identificar todos os arquivos que serão utilizados e realizar o carregamento dos dados que serão utilizados com uma remoção opcional do DC
def get_data_files(operations):
    files = []
    for op in operations:
        path = DATASET_PATH / op
        files.extend(list(path.glob("*.csv")))
    return files

def load_samples(file_path, remove_dc = False):
    try:
        data = np.genfromtxt(file_path, delimiter=",")
        #print(f"{file_path}: shape{data.shape}")
        if remove_dc:
            data = data - np.mean(data, axis=0)
        return data
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        return None

# Funções responsáveis pelos gráficos comparativos que serão gerados entre as informações normais e as anomalias, através de gráficos de linha, dispersão em 3D
def plot_comparison(normal_file, anomaly_file, remove_dc=False):
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
    return fig

def plot_3d_scatter(normal_files, anomaly_files, num_samples=3, feature_type="raw"):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    normal_data = []
    anomaly_data = []

    for i in range(min(num_samples, len(normal_files))):
        normal_sample =load_samples(normal_files[i], remove_dc=(feature_type == "raw"))
        anomaly_sample =load_samples(anomaly_files[i], remove_dc=(feature_type == "raw"))

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
            normal_data.append(np.square(normal_sample))
            anomaly_data.append(np.square(anomaly_sample))
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

    return fig

def plot_fft(df, st, sampling_rate=100, eixo='x', width=3.5, height=2.5):
    """
    Plota o espectro de frequência (FFT) de um eixo específico.

    Args:
        df (pd.DataFrame): DataFrame com colunas ['x', 'y', 'z']
        st (streamlit module): Módulo streamlit
        eixo (str): Eixo a ser plotado ('x', 'y' ou 'z')
        width (float): Largura da figura
        height (float): Altura da figura
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

    '''

    for axis in ["x", "y", "z"]:
        if axis in df.columns:
            signal = df[axis].values
            fft_values = np.abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal))

            fig, ax = plt.subplots()
            ax.plot(freqs[:len(freqs)//2], fft_values[:len(fft_values)//2])
            ax.set_title(f"FFT - Eixo {axis}")
            ax.set_xlabel("Frequência")
            ax.set_ylabel("Magnitude")
            st.pyplot(fig)

            plt.close(fig)
    '''
def plot_feature_histogram(features, st):
    for col in features.columns:
        fig, ax = plt.subplots()
        ax.hist(features[col], bins=20, color='skyblue', edgecolor="black")
        ax.set_title(f"Histograma - {col}")
        st.pyplot(fig)

        plt.close(fig)

def plot_histograms(normal_files, anomaly_files):
    
    #for f in normal_files:
    #    print(f"Verificando arquivo: {f}")
    #    print(f"Existe? {Path(f).exists()}")

    plt.figure(figsize=(12, 6))

    #normal_valid = [s for s in normal_files if isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[1] >= 3]
    #anomaly_valid = [s for s in anomaly_files if isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[1] >= 3]
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

    print(f'Amostras normais válidas: {len(normal_valid)}')
    print(f'Amostras anormais válidas: {len(anomaly_valid)}')

    if not normal_valid or not anomaly_valid:
        raise ValueError("Não há amostras válidas para plotar")
    
    num_features = normal_valid[0].shape[1]
    axis_labels = ["X-axis", "Y-axis", "Z-axis"]
    #num_features = len(axis_labels)
    
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
    if matplotlib.get_backend().lower() != 'agg':
        plt.show()

# Definição das estatísticas básicas que serão utilizadas como critérios de avaliação de desempenho dos algoritmos de aprendizado, são elas Média, variância, Kurtosis, Skew, MAD e correlação

def analyze_statistics(sample_file):
    sample = load_samples(sample_file, remove_dc=True)

    stats_dict = {
        "Sample shape": sample.shape, # Retorna as dimensões do conjunto de dados
        "Mean": np.mean(sample, axis=0), # retorna a média de cada eixo, para o caso, o valor médio da aceleração nos eixos
        "Variance": np.var(sample, axis=0), # Mede a dispersão dos dados em torno da média
        "Kurtosis": stats.kurtosis(sample), # Indica o achatamento da distribuição de dados
        "Skew": stats.skew(sample), # Skew, também conhecido como assimetria, indica a simetria da distribuição dos dados
        "MAD": stats.median_abs_deviation(sample), # Determina a mediana das diferenças absolutas entre os valores e a mediana dos dados
        "Correlation": np.corrcoef(sample.T), # Obtem a relação linear entre os diferentes eixos (X, Y, Z) do acelerômetro
    }

    return stats_dict
        
# Geração de um gráfico comparativo das frequências entre operações normais e anômalas, ideal para identificar padrões de vibração atípicos

def plot_fft_comparison(normal_files, anomaly_files, num_samples=200, start_bin=1, save_path=None):
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
        try:
            # Garante que a pasta existe
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # Tenta salvar a figura
            fig.savefig(str(save_path), dpi=300, bbox_inches = "tight")

            # Verifica se o arquivo foi salvo
            if Path(save_path).exists():
                print(f"[SUCESSO] Imagem salva em: {save_path.resolve()}")
            else:
                print(f"[ERRO] fig.savefig() executou mas o arquivo nao apareceu!")
        
        except Exception as e:
            print(f"[ERRO] Não foi possível salvar a imagem: {e}")

    #if save_path is not None:    
    #   fig.savefig(str(save_path), dpi=300, bbox_inches = "tight")
    #   print(f"[INFO] FFT comparison salva em: {save_path}")
       
    #plt.show()
    return fig

# --------------- ##

def find_optimal_threshold(normal_dist, anomaly_dist, n_splits=5):
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

            score = tp_rate - (5 * fp_rate)

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
    
def validate_model(normal_distances, anomaly_distances, threshold):
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
    
def plot_distance_distributions(normal_dist, anomaly_dist, threshold=None):
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
    #plt.show()
    
    return fig

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

    if save_path:
        plt.savefig(save_path)
        plt.close()
        return fig
    else:
        return fig
        #plt.show()
    

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pd.DataFrame(cm, index=["Normal", "Anomaly"], columns=["Normal", "Anomaly"]),
                     annot=True, fmt="d", cmap="Blues",
    )
    
    ax.set_title("Matriz de Confusão")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    #plt.show()

    return fig

def load_distributions():
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Caminho para salvar a imagem
    output_path = Path(f"Imagens/fft_comparison_{timestamp}.png")

    # Verificando se a pasta "Imagens" existe
    os.makedirs(output_path.parent, exist_ok=True)

    # Indicando o número de arquivos encontrados
    print(f"Amostras normais encontradas: {len(normal_files)}")
    print(f"Amostras anormais encontradas: {len(anomaly_files)}")

    # Gerando e salvando as comparações de plot com DC e sem DC
    plot_comparison(normal_files[0], anomaly_files[0], remove_dc=False)
    plot_comparison(normal_files[0], anomaly_files[0], remove_dc=True)

    # Gerando e salvando outros gráficos para análises (3D e histogramas)
    plot_3d_scatter(normal_files, anomaly_files, num_samples=10, feature_type='raw')
    plot_3d_scatter(normal_files, anomaly_files, num_samples=200, feature_type='raw')
    plot_3d_scatter(normal_files, anomaly_files, num_samples=200, feature_type='mean')
    plot_3d_scatter(normal_files, anomaly_files, num_samples=200, feature_type='variance')
    plot_3d_scatter(normal_files, anomaly_files, num_samples=200, feature_type='kurtosis')
    plot_3d_scatter(normal_files, anomaly_files, num_samples=200, feature_type='entropy')
    plot_3d_scatter(normal_files, anomaly_files, num_samples=200, feature_type='energy')

    # Gerando e mostrando histogramas
    plot_histograms(normal_files, anomaly_files)

    # Analisando estatísticas e imprimindo os resultados
    stat_results = analyze_statistics(normal_files[0])
    for key, value in stat_results.items():
        print(f"{key}:")
        print(value)
        print()

    # Gerando e salvando os gráficos de FFT (comparação)
    plot_fft_comparison(normal_files, anomaly_files, save_path=output_path)


    plot_distance_distributions(normal_dist, anomaly_dist, threshold)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(normal_dist, anomaly_dist)
    
    plt.show()

    # Confirmando que a imagem foi salva
    print(f"Comparação de FFt salva em: {output_path.resolve()}")

## ------------ ##

if __name__ == "__main__":
    main()
