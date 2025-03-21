# Panduan Menjalankan API Prediksi dengan FastAPI

## 📌 Persyaratan
Pastikan Anda telah menginstal Python dan pip. Jika belum, silakan unduh dan instal dari [python.org](https://www.python.org/).

## 📥 Instalasi Dependensi
Jalankan perintah berikut untuk menginstal dependensi yang diperlukan:
```sh
pip install fastapi uvicorn tensorflow pillow numpy
```

## 🚀 Menjalankan API
Simpan skrip FastAPI di file Python, misalnya `fastapi_model_api.py`, lalu jalankan perintah berikut:
```sh
uvicorn py.fastapi_model_api:app --reload
```
Setelah itu, API akan berjalan di `http://127.0.0.1:8000`.

## 📄 Menggunakan API
### 1️⃣ Melalui Browser
Buka dokumentasi interaktif API di:
👉 `http://127.0.0.1:8000/docs`

### 2️⃣ Menggunakan cURL
Gunakan cURL untuk mengirim gambar dan mendapatkan prediksi:
```sh
curl -X 'POST' 'http://127.0.0.1:8000/predict' -F 'file=@gambar.jpg'
```

### 3️⃣ Menggunakan Python (requests)
```python
import requests

url = "http://127.0.0.1:8000/predict"
files = {"file": open("gambar.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## 🎯 Output yang Diharapkan
Respon dari API akan berupa JSON dengan prediksi kelas dan probabilitasnya:
```json
{
  "class": "Normal",
  "confidence": 0.98
}
```

## 🛑 Menghentikan Server
Tekan `CTRL + C` di terminal untuk menghentikan server.