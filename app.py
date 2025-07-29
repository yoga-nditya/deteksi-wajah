from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
import os
import requests
from io import BytesIO

app = Flask(__name__)

DROPBOX_URL = "https://www.dropbox.com/scl/fi/d5uh5lj4rnliizq3lnr3h/model.h5?rlkey=xstzuu10lglb6ym8d5kcyvp38&st=pbwlefg7&dl=1"
MODEL_WEIGHTS = "model.h5"

# --- URUTAN LABEL HARUS SESUAI TRAINING ---
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def grayscale_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)

# Definisikan arsitektur model sesuai pipeline
inputs = Input(shape=(48, 48, 1), name='grayscale_input')
x = Lambda(grayscale_to_rgb, name='grayscale_to_rgb')(inputs)

x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1')(x)
x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2,2), name='block1_pool')(x)
x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2,2), name='block2_pool')(x)
x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2,2), name='block3_pool')(x)
x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2,2), name='block4_pool')(x)
x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2,2), name='block5_pool')(x)
x = GlobalAveragePooling2D(name='global_avg_pool')(x)
x = Dense(256, activation='relu', name='fc1')(x)
x = BatchNormalization(name='bn1')(x)
x = Dropout(0.5, name='dropout1')(x)
outputs = Dense(len(CLASS_NAMES), activation='softmax', name='predictions')(x)

model = Model(inputs, outputs)
print("✅ Model architecture loaded!")

# Download weights jika belum ada
if not os.path.exists(MODEL_WEIGHTS):
    print("📥 Downloading weights from Dropbox ...")
    with requests.get(DROPBOX_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_WEIGHTS, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"✅ Weights downloaded: {MODEL_WEIGHTS}")

# Load weights
model.load_weights(MODEL_WEIGHTS)
print("✅ Model weights loaded!")

@app.route('/')
def home():
    return "API OK! Model loaded!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image and preprocess
        img = image.load_img(BytesIO(file.read()), color_mode='grayscale', target_size=(48, 48))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        preds = model.predict(x)[0]  # (7,)
        pred_idx = int(np.argmax(preds))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = round(float(np.max(preds)) * 100, 1)

        # Top 3 predictions
        top3_idx = np.argsort(preds)[::-1][:3]
        top_predictions = [
            {
                "label": CLASS_NAMES[idx],
                "confidence": round(float(preds[idx]) * 100, 1)
            }
            for idx in top3_idx
        ]

        return jsonify({
            'predicted_emotion': pred_label,
            'confidence': confidence,
            'top_predictions': top_predictions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get("PORT", 5000))
    )

