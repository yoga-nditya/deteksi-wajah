import tensorflow as tf
import requests

# Download model dari Dropbox
DROPBOX_URL = "https://www.dropbox.com/scl/fi/l3d9nprrb4un4km2r13l6/best_vgg16_balanced_strategy6_1-2.h5?rlkey=qd9jrm3hh6p7j0dyvqocichx7&st=tqa9ligf&dl=1"
MODEL_LAMA = "best_vgg16_balanced_strategy6_1-2.h5"
MODEL_BARU = "model_no_lambda.h5"

# Download model H5 dari Dropbox (jika belum ada)
if not tf.io.gfile.exists(MODEL_LAMA):
    print("📥 Downloading model from Dropbox ...")
    r = requests.get(DROPBOX_URL, stream=True)
    with open(MODEL_LAMA, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk: f.write(chunk)
    print(f"✅ Model downloaded: {MODEL_LAMA}")

# Fungsi Lambda dari model aslinya (untuk load)
def grayscale_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)

# Load model ASLI dengan Lambda
model_lama = tf.keras.models.load_model(
    MODEL_LAMA, custom_objects={"grayscale_to_rgb": grayscale_to_rgb}
)
print("✅ Model asli (dengan Lambda) berhasil diload.")

# Buat model BARU tanpa Lambda Layer
from tensorflow.keras import Input, Model

# INPUT BARU: (48,48,3)
inputs = Input((48, 48, 3), name="input_rgb")
x = inputs
# SKIP: Input(0), Lambda(1). Mulai dari layer index 2.
for layer in model_lama.layers[2:]:
    x = layer(x)
model_baru = Model(inputs, x)

# Save model baru tanpa Lambda
model_baru.save(MODEL_BARU)
print(f"✅ Model tanpa Lambda berhasil disimpan: {MODEL_BARU}")
