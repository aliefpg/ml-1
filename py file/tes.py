# Import necessary libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model(r'model/model_mobilenet.keras')

# Define class labels
class_names = ['air', 'anggur', 'apel']

# Function to predict images from multiple folders
def predict_from_multiple_folders(folder_paths, threshold=0.7):
    for folder_path in folder_paths:
        print(f"\n=== Testing Images from: {folder_path} ===")
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} tidak ditemukan!\n")
            continue
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            # Load and preprocess the image (gunakan ukuran 224x224 agar sama dengan training)
            img = image.load_img(img_path, target_size=(224, 224))  
            img_array = image.img_to_array(img) / 255.0  
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction[0])  
            confidence = np.max(prediction[0])  

            # Gunakan threshold untuk memastikan prediksi cukup yakin
            if confidence >= threshold:
                predicted_label = class_names[predicted_class]
            else:
                predicted_label = "uncertain"

            # Display the prediction result
            print(f"Image: {img_name} => Predicted: {predicted_label} with confidence {confidence:.2f}")
        print('-' * 50)

# Define test folder paths
test_folders = [
    'dataset/image/test/apel',
    'dataset/image/test/anggur',
    'dataset/image/test/air'
]

# Run predictions
predict_from_multiple_folders(test_folders, threshold=0.7)
