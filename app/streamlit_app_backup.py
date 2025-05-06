import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from pathlib import Path


from ProjetoSensor.esp32.training import extract_ml_features, preprocess_features, clip_features
from ProjetoSensor.esp32.analysis import plot_fft, plot_feature_histogram
from app.utils import check_anomaly, adjust_threshold

# Baseline
DATA_PATH = Path("ProjetoSensor/datasets/ac/latest_data/baseline_latest_data.csv")
#DATA_PATH = Path("ProjetoSensor/datasets/ac/latest_data/baseline_silent_1_latest_data.csv")

# Anomaly
#DATA_PATH = Path("ProjetoSensor/datasets/ac/latest_data/anomaly_latest_data.csv")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mahalonobis_model.npz"
THRESHOLD_PATH = Path(__file__).resolve().parent.parent / "models" / "threshold.joblib"
SCALER_PATH = Path(__file__).resolve().parent.parent / "models" / "scaler.pkl"

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
def load_model(data_path):
    data = np.load(data_path, allow_pickle=True)
    mu = data['mu']
    cov = data['cov']
    threshold = data['threshold']
    keys = list(data.keys())
    expected_keys = {"mu", "cov", "threshold"}
    #scaler = data['scaler'].item()
    #return data["mu"], data["cov"], data["threshold"], data["scaler"].item()
    #return data["mu"], data["cov"], data["threshold"]
    if not expected_keys.issubset(keys):
        st.error(f"Modelo incompleto. Esperado: {expected_keys}, Encontrado: {keys}")
    return mu, cov, threshold

@st.cache_resource
def load_scaler(scaler_path):
    scaler = joblib.load(scaler_path)
    print("LOAD_SCALER: Scaler center (medians):", getattr(scaler, 'center_', 'Not fitted'))
    print("LOAD_SCALER: Scaler center (IQR):", getattr(scaler, 'scale_', 'Not fitted'))
    return scaler

try:
    mu, cov, threshold = load_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
    print("TRY: Scaler center (medians):", getattr(scaler, 'center_', 'Not fitted'))
    print("TRY: Scaler center (IQR):", getattr(scaler, 'scale_', 'Not fitted'))
    #print(f"Tipo de scaler carregado: {type(scaler)}")
    #scaler = StandardScaler()
    #print(f"Scaler: {scaler}")
    if not isinstance(scaler, RobustScaler):
        st.error(f"Scaler não carregado corretamente!! Esperado: RobustScaler, Encontrado: {type(scaler)}")
    #else:
        #st.write("Scaler carregado corretamente!")
    print(f"Scaler: {scaler}")
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
        print("Scaler foi treinado com: ", scaler.n_features_in_, "features")
        features_raw = df[["x", "y", "z"]].values
        print("Shape do features_raw:", features_raw.shape)
        features = extract_ml_features(features_raw)
        #features = remove_low_iqr_features(features)
        #features = preprocess_features(features)
        features = np.clip(features, -10, 10)
       
        #st.write("Shape das features extraídas:", features.shape)
        #st.write("Features (sem scaler):", features)

        #scaler.fit([features])
        
        #features = np.atleast_2d(features)
        #features = features.reshape(1, -1)
        print("Shape do features_raw antes do 2d:", features.shape)
        print("STREAMLIT SEM SCALER: ", features)

        #features_scaled = scaler.transform(features)
        print("Scaler center (medians):", getattr(scaler, 'center_', 'Not fitted'))
        print("Scaler center (IQR):", getattr(scaler, 'scale_', 'Not fitted'))
        features_scaled = scaler.transform(np.atleast_2d(features))
        print("STREAMLIT COM SCALER: ", features_scaled)
        #features_scaled = remove_outliers_by_iqr(features_scaled, threshold)

    
       

        #features = extract_ml_features(df[["x", "y","z"]].values)
        #print("Features extraidas: ", features)
        #features = np.atleast_2d(features)
        #print("Features 2D: ", features)
        #features = scaler.transform(features)
        #print("Features com scaler: ", features)
        #st.write("shape das features: ", features.shape)
        #st.write(features)


    except Exception as e:
        st.error(f"Erro na extração de features: {e}")
        features = None
        features_scaled = None
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
    if mu is None or cov is None or threshold is None:
        print(f"Mu: {mu}\n cov: {cov}\n threshold: {threshold}")
        st.error("Erro: média ou matriz de covariância ou threshold não carregadas.")
                
        #st.warning("Modelo não carregado corretamente. Verifique os arquivos.")
        
    else:
        #cov_inv = np.linalg.inv(cov)
        #threshold = adjust_threshold(features, mu, cov)
        dist, is_anomaly = check_anomaly(features_scaled, mu, cov, threshold)

        st.metric("Distância de Mahalanobis", f"{dist[0]:.2f}")
        st.metric("Threshold: ", f"{threshold:.2f}")

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
    st.warning(f"Arquivo não encontrado: {DATA_PATH}")