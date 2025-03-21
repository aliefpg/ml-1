**Project Dependencies & Structure**

### 1. Bahasa & Framework
- **Python** (versi 3.11)
- **FastAPI** (untuk API)
- **TensorFlow/Keras** (untuk deep learning model)
- **Uvicorn** (untuk menjalankan FastAPI)
- **Pandas** (untuk membaca data harga)
- **Pillow** (untuk memproses gambar)

### 2. Struktur Folder
```
/py
│
├── /dataset
│   ├── /image
│   │   ├── /train
│   │   ├── /test
│   │   ├── /valid
│   ├── data_harga.csv
│
├── /model
│   ├── model_mobilenet.keras
│
├── class.py  (Data Augmentation & Data Loader)
├── fastapi_model_api.py  (API untuk Prediksi Harga)
├── harga.py  (Prediksi dari CSV)
├── tes.py  (Testing model pada folder dataset)
├── train.py  (Melatih model menggunakan MobileNetV2)
└── requirements.txt  (Daftar dependencies)
```

### 3. Library yang Dipakai
- `tensorflow`
- `fastapi`
- `uvicorn`
- `numpy`
- `pandas`
- `pillow`
- `os`

### 4. Cara Menjalankan
1. **Instal dependensi**
   ```sh
   pip install -r requirements.txt
   ```
2. **Mengecek class**
   ```sh
   python class.py
   ```
3. **Menjalankan training model**
   ```sh
   python train.py
   ```
4. **Menguji model pada dataset**
   ```sh
   python tes.py
   ```
5. **Mencocokan harga dengan gambar**
   ```sh
   python harga.py
   ```
6. **Menjalankan API FastAPI**
   ```sh
  uvicorn py.fastapi_model_api:app --reload
   ```
