import io
import gdown
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Google Drive model link (direct)
MODEL_URL = 'https://drive.google.com/file/d/1zG0YYVqZK7BrvHtTdZ-ht4GQD_JERPva/view?usp=sharing'
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

model = None

# Lambda: grayscale to RGB
@tf.keras.utils.register_keras_serializable(package='Custom')
def grayscale_to_rgb(x):
    return tf.repeat(x, 3, axis=-1)

def download_and_load_model():
    global model
    print("📥 Downloading model from Google Drive (in-memory)...")
    # Download as bytes (no file saved to disk)
    response = gdown.download(MODEL_URL, quiet=False, fuzzy=True)
    
    # Read from downloaded file path directly into memory
    model = load_model(
        response,
        custom_objects={
            'grayscale_to_rgb': grayscale_to_rgb,
            'Lambda': Lambda(grayscale_to_rgb, output_shape=(None, 48, 48, 3))
        },
        compile=False
    )
    print("✅ Model loaded successfully from memory!")

def preprocess_image(image_file, target_size=(48, 48)):
    image = Image.open(image_file).convert('L')  # Grayscale
    image = image.resize(target_size)
    image = np.asarray(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)  # (H, W, 1)
    image = np.expand_dims(image, axis=0)   # (1, H, W, 1)
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

        # Top 3 predictions
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
    download_and_load_model()  # Load from GDrive every time on start
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
