import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# 1. Dataset Paths
train_dir = "dataset/image/train"
valid_dir = "dataset/image/valid"

# 2. Data Augmentasi
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# 3. Load Data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=10,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 4. Load Pretrained Model (MobileNetV2) dengan perbaikan input_shape
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Bekukan semua layer awal

# 5. Bangun Model dengan Fully Connected Layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# 6. Compile Model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 7. Training Model
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=5
)

# 8. Simpan Model
model.save("model/model_mobilenet2.keras")

# 9. Ambil akurasi akhir dari training dan validasi
final_train_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100

print("\n===================================")
print(f"✅ Training Selesai!")
print(f"🎯 Akurasi akhir training: {final_train_acc:.2f}%")
print(f"🎯 Akurasi akhir validasi: {final_val_acc:.2f}%")
print("✅ Model berhasil disimpan sebagai model_mobilenet.keras")
print("===================================\n")