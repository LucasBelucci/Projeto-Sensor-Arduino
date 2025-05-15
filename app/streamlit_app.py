import streamlit as st
import pandas as pd
from pathlib import Path
from utils import AnomalyDetector

# Baseline
DATA_PATH = Path("ProjetoSensor/datasets/ac/latest_data/baseline_latest_data.csv")
#DATA_PATH = Path("ProjetoSensor/datasets/ac/latest_data/baseline_silent_1_latest_data.csv")

# Anomaly
#DATA_PATH = Path("ProjetoSensor/datasets/ac/latest_data/anomaly_latest_data.csv")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mahalonobis_model.npz"
#THRESHOLD_PATH = Path(__file__).resolve().parent.parent / "models" / "threshold.joblib"
#SCALER_PATH = Path(__file__).resolve().parent.parent / "models" / "scaler.pkl"

st.title("Detecção de Anomalias com acelerômetro")

if "detector" not in st.session_state:
    st.session_state.detector = AnomalyDetector(MODEL_PATH)

uploaded_file = st.file_uploader("Envie um arquivo CSV com os dados do sensor")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Prévia dos dados: ", df.head())

    data = df.to_numpy()

    if st.button("Detectar Anomalia"):
        result = st.session_state.detector.predict(data)
        st.json(result)
        
        st.write("Últimas distâncias calculadas:")
        st.write(st.session_state.detector.recent_distances)
        st.metric("Anomalia", "Sim" if result["is_anomaly"] else "Não")
        st.metric("Confiança", f"{result['confidence']*100:.1f}%")
        st.metric("Distância", f"{result['distance']:.2f}")