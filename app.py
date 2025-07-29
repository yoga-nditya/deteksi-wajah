import os
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

DROPBOX_URL = "https://www.dropbox.com/scl/fi/i9vckqfq5fz3twoxhdr3z/dowload.h5?rlkey=8dtnvaxx2330ouuuykxffzv8d&dl=1"
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

model = None

@tf.keras.utils.register_keras_serializable(package='Custom')
def grayscale_to_rgb(x):
    return tf.repeat(x, 3, axis=-1)

def download_and_load_model():
    global model
    output = 'model.h5'
    if not os.path.exists(output):  # biar gak download ulang tiap restart
        print("📥 Downloading model from Dropbox ...")
        r = requests.get(DROPBOX_URL, stream=True)
        r.raise_for_status()
        with open(output, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("[INFO] Model file size:", os.path.getsize(output), "bytes")
    model = load_model(
        output,
        custom_objects={'grayscale_to_rgb': grayscale_to_rgb},
        compile=False
    )
    print("✅ Model loaded successfully from file:", output)

def preprocess_image(image_file, target_size=(48, 48)):
    image = Image.open(image_file).convert('L')
    image = image.resize(target_size)
    image = np.asarray(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model not loaded yet'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    try:
        image = preprocess_image(image_file)
        predictions = model.predict(image)[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index]) * 100

        top3_indices = np.argsort(predictions)[-3:][::-1]
        top3 = [{
            'label': CLASS_NAMES[i],
            'confidence': round(float(predictions[i]) * 100, 2)
        } for i in top3_indices]

        return jsonify({
            'predicted_class': CLASS_NAMES[predicted_index],
            'confidence': round(confidence, 2),
            'top_predictions': top3
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Emotion detection API is running'
    })

if __name__ == '__main__':
    download_and_load_model()
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
