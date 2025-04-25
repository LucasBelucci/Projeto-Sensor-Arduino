import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


from ProjetoSensor.esp32.training import extract_ml_features
from ProjetoSensor.esp32.analysis import plot_fft, plot_feature_histogram
from app.utils import check_anomaly

# Baseline
DATA_PATH = Path("ProjetoSensor/datasets/ac/latest_data/baseline_latest_data.csv")

# Anomaly
#DATA_PATH = Path("ProjetoSensor/datasets/ac/latest_data/anomaly_latest_data.csv")


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mahalonobis_model.npz"

# Setup app

st.set_page_config("Detecção de anomalias", layout = "centered")
st.title("Monitoramente de Vibração - ESP32")

# Mudando a cor de fundo
#st.markdown("""
#    <style>
#            body {
#                background-color: #f4f4f9;
#            }
#            .main {
#                background-color: #ffffff;
#            }
#    </style>
#""", unsafe_allow_html=True)

# Carregando o modelo
@st.cache_resource
def load_model(path):
    data = np.load(path, allow_pickle=True)
    return data["mu"], data["cov"], data["threshold"], data["scaler"].item()

try:
    mu, cov, threshold, scaler = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")

# Atualizar os dados
if st.button("Atualizar dados"):
    st.rerun()

# Exibindo os resultados
st.subheader("Ultimos dados do acelerômetro")
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH, header=None, names=["x", "y", "z"])
    st.dataframe(df.tail(10))

    # Exibir espectro FFT
    st.subheader("Espectro de Frequência (FFT)")
    
    cols = st.columns(3)
    for eixo, col in zip(['x', 'y', 'z'], cols):
        with col:
            plot_fft(df, st, eixo=eixo, width=3.5, height=2.5)

    #col1, col2, col3 = st.columns(3)
    
    #with col1:
    #    plot_fft(df, st) # FFT eixo X
    
    #with col2:
    #    plot_fft(df, st) # FFT eixo Y
    
    #with col3:
    #    plot_fft(df, st) # FFT eixo Z

    # Extração de features
    #st.subheader("Features extraídas")
    try:
        features = extract_ml_features(df[["x", "y","z"]].values)
        #st.write("shape das features: ", features.shape)
        #st.write(features)
    except Exception as e:
        st.error(f"Erro na extração de features: {e}")
        features = None
    #st.write(features)


    # Plotas histogramas - Interessante, mas ficou muito poluído
    #st.subheader("Distribuição das Features")
    #feature_names = [
    # Tempo - estatísticas
    #"mean_x", "mean_y", "mean_z",
    #"var_x", "var_y", "var_z",
    #"skew_x", "skew_y", "skew_z",
    #"kurtosis_x", "kurtosis_y", "kurtosis_z",
    #"mad_x", "mad_y", "mad_z",

    # Correlação entre eixos
    #"corr_xy", "corr_xz", "corr_yz",

    # FFT - estatísticas
    #"fft_mean_x", "fft_mean_y", "fft_mean_z",
    #"fft_std_x", "fft_std_y", "fft_std_z",
    #"fft_energy_x", "fft_energy_y", "fft_energy_z",
    #"fft_peak_x", "fft_peak_y", "fft_peak_z"
    #]
    #features_df = pd.DataFrame([features], columns=feature_names)
    #plot_feature_histogram(features_df, st)


    # Verificação da anomalia
    st.subheader("Detecção de Anomalia")
    if 'mu' in locals() and 'cov' in locals() and 'threshold' in locals():
        cov_inv = np.linalg.inv(cov)
        dist, is_anomaly = check_anomaly(features, mu, cov_inv, threshold)

        st.metric("Distância de Mahalanobis", f"{dist[0]:.2f}")

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Destaque da Anomalia
        if is_anomaly:
            st.markdown("<div style='background-color:#ffcccc;padding:10px;border-radius:10px;'>"
                "<h4 style='color:red;'>🚨 Anomalia detectada!</h4></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background-color:#ccffcc;padding:10px;border-radius:10px;'>"
                "<h4 style='color:green;'>✅ Nenhuma anomalia detectada.</h4></div>", unsafe_allow_html=True)
        #st.metric("É anomalia?", "Sim" if is_anomaly else "Não")
    else:
        st.warning("Modelo não carregado corretamente. Verifique os arquivos.")

else:
    st.warning(f"Arquivo não encontrado: {DATA_PATH}")