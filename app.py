from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)

# Definisikan ulang fungsi lambda sesuai model
def grayscale_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)

# Load model .h5 Keras dengan custom_objects
try:
    model = load_model('model.h5', custom_objects={'grayscale_to_rgb': grayscale_to_rgb})
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ ERROR loading model:", str(e))
    exit(1)

@app.route('/')
def home():
    return "API OK! Model loaded!"

# Contoh endpoint prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Preprocess sesuai kebutuhan model, contoh (ubah sesuai model kamu):
    img = image.load_img(file, color_mode='grayscale', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)         # shape (1, 224, 224, 1)
    x = x / 255.0                         # Normalisasi (jika diperlukan)

    # Prediksi
    preds = model.predict(x)
    label = np.argmax(preds[0])

    return jsonify({'result': int(label), 'confidence': float(np.max(preds[0]))})

if __name__ == '__main__':
    app.run(debug=True)
