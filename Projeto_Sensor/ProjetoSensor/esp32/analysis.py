# %% Setup
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
7
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
    data = np.genfromtxt(file_path, delimiter=",")
    if remove_dc:
        data = data - np.mean(data, axis=0)
    return data

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
        else:
            normal_data.append(normal_sample)
            anomaly_data.append(anomaly_sample)

    if feature_type in ['mean', 'variance', 'kurtosis']:
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

def extract_fft_features(sample):
    hann_widow = np.hanning(sample.shape[0])

    out_sample = np.zeros((int(sample.shape[0]/2),  sample.shape[1]))
    for i, axis in enumerate(sample.T):
        fft = abs(np.fft.rfft(axis*hann_widow))
        out_sample[:, i] = fft[1:]

    return out_sample

def plot_fft_comparison(normal_files, anomaly_files, num_samples=200, start_bin=1):
    normal_ffts = []
    anomaly_ffts = []

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
    return fig

# %%

normal_files = get_data_files(NORMAL_OPS)
anomaly_files = get_data_files(ANOMALY_OPS)

print(f"Found {len(normal_files)} normal operation files")
print(f"Found {len(anomaly_files)} anomaly operation files")

plot_comparison(normal_files[0], anomaly_files[0], remove_dc=False)
plot_comparison(normal_files[0], anomaly_files[0], remove_dc=True)

plot_3d_scatter(normal_files, anomaly_files, num_samples=10, feature_type='raw')
plot_3d_scatter(normal_files, anomaly_files, num_samples=200, feature_type='raw')
plot_3d_scatter(normal_files, anomaly_files, num_samples=200, feature_type='mean')
plot_3d_scatter(normal_files, anomaly_files, num_samples=200, feature_type='variance')
plot_3d_scatter(normal_files, anomaly_files, num_samples=200, feature_type='kurtosis')


stat_results = analyze_statistics(normal_files[0])
for key, value in stat_results.items():
    print(f"{key}:")
    print(value)
    print()

plot_fft_comparison(normal_files, anomaly_files)
# %%
