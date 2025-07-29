from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os
import requests

app = Flask(__name__)

# Link model dari Dropbox (HARUS model TANPA LAMBDA LAYER, input (48,48,3))
DROPBOX_URL = "https://www.dropbox.com/scl/fi/d5uh5lj4rnliizq3lnr3h/model.h5?rlkey=xstzuu10lglb6ym8d5kcyvp38&st=pbwlefg7&dl=1"
MODEL_PATH = "model_no_lambda.h5"

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

# Download dan load model (TANPA Lambda Layer!)
try:
    download_model()
    model = load_model(MODEL_PATH)
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

    # Preprocess gambar: grayscale→RGB (48,48,3)
    img = image.load_img(file, color_mode='grayscale', target_size=(48, 48))
    x = image.img_to_array(img)
    x = tf.image.grayscale_to_rgb(x)    # Convert ke RGB, hasil (48,48,3)
    x = np.expand_dims(x, axis=0)       # shape (1, 48, 48, 3)
    x = x / 255.0                       # Normalisasi

    preds = model.predict(x)
    label = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    return jsonify({'result': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
