from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os
import requests

app = Flask(__name__)

# Link model dari Dropbox, WAJIB pakai dl=1!
DROPBOX_URL = "https://www.dropbox.com/scl/fi/d5uh5lj4rnliizq3lnr3h/model.h5?rlkey=xstzuu10lglb6ym8d5kcyvp38&st=pbwlefg7&dl=1"
MODEL_PATH = "model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Dropbox ...")
        resp = requests.get(DROPBOX_URL, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Model downloaded! Size: {os.path.getsize(MODEL_PATH)} bytes")
        # Magic bytes check
        with open(MODEL_PATH, "rb") as f:
            print("Magic bytes:", f.read(8))

# Lambda function sesuai dengan model Keras Lambda layer
def grayscale_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)

# Download dan load model
try:
    download_model()
    model = load_model(MODEL_PATH, custom_objects={'grayscale_to_rgb': grayscale_to_rgb})
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ ERROR loading model:", str(e))
    exit(1)

@app.route('/')
def home():
    return "API OK! Model loaded!"

# Endpoint prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Preprocess sesuai kebutuhan model
    img = image.load_img(file, color_mode='grayscale', target_size=(48, 48))  # Ganti ukuran sesuai model!
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)         # shape (1, 48, 48, 1)
    x = x / 255.0                         # Normalisasi (jika diperlukan)

    preds = model.predict(x)
    label = int(np.argmax(preds[0]))

    return jsonify({'result': label, 'confidence': float(np.max(preds[0]))})

if __name__ == '__main__':
    app.run(debug=True)
