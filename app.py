from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os

app = Flask(__name__)

# Load Model
model = load_model("asl_model.h5")

# Define Classes
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        image_data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image. Please upload a valid image file."}), 400

        # Resize and normalize image
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img)
        predicted_class = classes[np.argmax(prediction)]

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
