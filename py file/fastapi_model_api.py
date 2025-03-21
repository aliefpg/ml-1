from fastapi import FastAPI, File, UploadFile
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import uvicorn
import os
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load model yang sudah dilatih
model = load_model('model_mobilenet.keras')

# Load daftar kelas dari training
class_indices = {'air': 0, 'anggur': 1, 'apel': 2}
class_labels = {v: k for k, v in class_indices.items()}

# Baca file CSV harga
df_harga = pd.read_csv('data_harga.csv', sep=";")
df_harga["nama"] = df_harga["nama"].str.lower()

# Fungsi untuk mendapatkan harga berdasarkan nama barang
def get_harga(nama_barang):
    harga = df_harga[df_harga['nama'] == nama_barang]['harga'].values
    return harga[0] if len(harga) > 0 else "Tidak ditemukan"

# Fungsi untuk memprediksi gambar
def prediksi_gambar(img: Image.Image):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediksi = model.predict(img_array)
    prediksi_label = np.argmax(prediksi)
    nama_barang = class_labels[prediksi_label]
    harga_barang = get_harga(nama_barang)
    
    return {"prediksi": nama_barang, "harga": f"Rp {harga_barang}"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))
        hasil_prediksi = prediksi_gambar(img)
        return {"filename": file.filename, **hasil_prediksi}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
