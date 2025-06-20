import numpy as np
from scipy import stats as scipy_stats  # Renamed to avoid shadowing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime
import logging
from pathlib import Path

# Configure logging

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        self.mu = data['mu']
        self.cov = data['cov']
        self.threshold = data['threshold']
        self.last_predictions = [False, False, False]
        self.recent_distances = []
        logger.info(
            "Modelo carregado com threshold: %.2f", self.threshold
        )

    def preprocess(self, data, remove_dc=True):
        if remove_dc:
            data = data - np.mean(data, axis=0)
        return data

    def extract_features(self, sample):
        features = []
        for axis_idx in range(sample.shape[1]):
            axis_data = sample[:, axis_idx]

            features.extend(
                [
                    np.std(axis_data),
                    scipy_stats.kurtosis(axis_data),
                    np.max(np.abs(axis_data)),
                    np.sqrt(np.mean(np.square(axis_data))),
                    np.max(axis_data) - np.min(axis_data),
                ]
            )
    
        return np.array(features)

    def mahalanobis_distance(self, x):
        x_mu = x - self.mu
        epsilon = 1e-6
        cov_reg = self.cov + epsilon * np.eye(self.cov.shape[0])

        try:
            scale = np.median(np.diag(cov_reg))
            cov_scaled = cov_reg / scale
            inv_covmat = np.linalg(cov_scaled) / scale

            if x_mu.ndim == 1:
                mahal = np.sqrt(np.dot(np.dot(x_mu, inv_covmat), x_mu))
        
            else:
                mahal = np.sqrt(np.sum(np.dot(x_mu, inv_covmat) * x_mu, axis=1))
            return mahal
        except np.linalg.LinAlgError:
            return np.inf

    def calculate_confidence(self, distance):
        self.recent_distances.append(distance)
        self.recent_distances = self.recent_distances[-20:]

        threshold_magnitude = np.log10(self.threshold)

        lower_band_factor = np.exp(-threshold_magnitude / 2)
        upper_band_factor = np.exp(threshold_magnitude / 2)

        lower_bound = self.threshold * lower_band_factor
        upper_bound = self.threshold * upper_band_factor

        if distance < lower_bound:
            base_confidence = 0.95
    
        elif distance > upper_bound:
            base_confidence = 0.90

        else:
            x = (distance - lower_bound) / (upper_bound - lower_bound)
            base_confidence = 0.9 / (1 + np.exp((x - 0.5) * 10))

        if len(self.recent_distances) > 5:
            recent_mean = np.mean(self.recent_distances[-5:])
            recent_std = np.std(self.recent_distances[-5:])

            trend_stability = np.exp(-abs(distance - recent_mean) / (recent_std + 1e-6))

            variation_coefficient = recent_std / (recent_mean + 1e-6)
            variation_stability = np.exp(-variation_coefficient)

            stability_factor = (trend_stability + variation_stability) / 2

            history_weight = min(len(self.recent_distances) / 20, 1.0)
            final_confidence = (
                base_confidence * (1 - history_weight) + (base_confidence + stability_factor) / 2 * history_weight
            )
        else:
            final_confidence = base_confidence

        return float(np.clip(final_confidence, 0.0, 1.0))

    def predict(self, data):
        processed_data = self.preprocess(data)
        features = self.extract_features(processed_data)
        distance = float(self.mahalanobis_distance(features))

        is_anomaly = distance > self.threshold

        self.last_predictions.pop(0)
        self.last_predictions.append(is_anomaly)

        stable_anomaly = sum(self.last_predictions) >= 2

        confidence = self.calculate_confidence(distance)

        features_name = [
            "std",
            "kurtosis",
            "peak_amplitude",
            "rms",
            "peak_to_peak",
        ]

        feature_stats = {}

        n_features_per_axis = len(features_name)
        n_axes = len(features) // n_features_per_axis

        for axis_idx in range(n_axes):
            start_idx= axis_idx * n_features_per_axis
            axis_features = features[start_idx : start_idx + n_features_per_axis]
            feature_stats[f"axis_{axis_idx}"] = {
                name: float(value) for name, value in zip(features_name, axis_features)
            }

        result = {
            "is_anomaly": bool(stable_anomaly),
            "confidence": float(confidence),
            "distance": float(distance),
            "threshold": float(self.threshold),
            "feature_values": feature_stats,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("=" * 50)
        logger.info("Prediction details:")
        logger.info("Timestamp: %s", result["timestamp"])
        logger.info("Is anomaly: %s", result["is_anomaly"])
        logger.info("Confidence: %.3f", result["confidence"])
        logger.info("Distance: %.3f (threshold: %.3f)", result["distance"], result["threshold"])
        logger.info("Features Values:")
        for axis_name, stats in feature_stats.items():
            logger.info("  %s:", axis_name)
            for feat, val in stats.items():
                logger.info("    %s: %.3f", feat, val)
        logger.info("=" * 50)

        return result
    
class AccelerometerData(BaseModel):
    data: List[List[float]]
    sensor_id: str = "default"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = AnomalyDetector(Path(__file__).resolve().parent.parent / "models" / "mahalonobis_model.npz")

@app.post("/predict")
async def predict_anomaly(data: AccelerometerData):
    try:
        array_data = np.array(data.data)
        logger.info(
            "Received data shape: %s from sensor %s", array_data.shape, data.sensor_id
        )

        result = detector.predict(array_data)
        return result
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")