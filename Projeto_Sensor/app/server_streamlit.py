from flask import Flask, request, jsonify
import json
import os
import csv


def get_secrets():
    with open('keys.json') as f:
        return json.load()

secrets = get_secrets()
DATA_DIR = secrets.get("DATABASE_URL_STREAMLIT")
DEBUG_MODE = secrets.get("DEBUG", False)

CSV_PATH = os.path.join(DATA_DIR, "latest_data.csv")
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def status():
    return "Servidor Flask ativo", 200

@app.route("/send-data", methods=["POST"])
def receive_data():
    try:
        data = request.get_json()

        if not all(axis in data for axis in ("x", "y", "z")):
            return jsonify({"erros": "Dados incompletos. Esperado: x, y, z"}), 400
        
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z"])
            for i in range(len(data["x"])):
                writer.writerow([data["x"][i], data["y"][i], data["z"][i]])

        return '', 204 # No Content
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=DEBUG_MODE)