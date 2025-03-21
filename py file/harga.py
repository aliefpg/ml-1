# Import Library
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# Load model yang sudah dilatih
model = load_model(r'model/model_mobilenet.keras')

# Load daftar kelas dari training
class_indices = {'air': 0, 'anggur': 1, 'apel': 2}
class_labels = {v: k for k, v in class_indices.items()}

# Baca file CSV harga
df_harga = pd.read_csv(r'dataset/data_harga.csv', sep=";")
df_harga["nama"] = df_harga["nama"].str.lower()

# Fungsi untuk mendapatkan harga berdasarkan nama barang
def get_harga(nama_barang):
    harga = df_harga[df_harga['nama'] == nama_barang]['harga'].values
    return harga[0] if len(harga) > 0 else "Tidak ditemukan"

# Fungsi untuk memprediksi gambar
def prediksi_gambar(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediksi = model.predict(img_array)
    prediksi_label = np.argmax(prediksi)
    nama_barang = class_labels[prediksi_label]

    harga_barang = get_harga(nama_barang)

    print(f"Gambar: {img_path}")
    print(f"Prediksi: {nama_barang}")
    print(f"Harga: Rp {harga_barang}")
    print("-" * 30)

# Ambil semua gambar di folder test tanpa XML
test_folder = 'dataset/image/test/'
for kategori in class_indices.keys():
    path_kategori = os.path.join(test_folder, kategori)
    if os.path.exists(path_kategori):
        for file in os.listdir(path_kategori):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(path_kategori, file)
                prediksi_gambar(img_path)
