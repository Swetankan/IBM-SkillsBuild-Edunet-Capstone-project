from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Heart Disease Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    input_features = np.array([data["features"]])
    prediction = model.predict(input_features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
